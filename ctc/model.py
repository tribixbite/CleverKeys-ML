#!/usr/bin/env python3
"""Layout-agnostic CTC swipe-emission encoder.

I/O contract (must match CleverKeys ``src/main/kotlin/.../swipe/ctc/`` exactly):
  in : features    [B, 2, 64]  float32   (x row, y row; [0,1])
       layout_keys [B, 64, 2]  float32   (key centers, pad = (0,0))
       layout_mask [B, 64]     bool      (true for real keys)
  out: log_emissions [B, 32, 65] float32 (log-softmaxed; blank = index 64)
       coefficients  [B, 32, 64] float32 (spatial coefficients, phase-2 input)
       lambda        [B, 32, 1]  float32 (per-frame positive gate)

Audit fix #2 (the rank defect)
------------------------------
The recipe scored keys by sampling a FIXED 8x8 cosine (DCT) field at the key
centers: ``key_logits = einsum(coeff, cos_basis) * lambda``. At the 26 canonical
en_qwerty centers that basis matrix has **rank 23, not 26** (singular values
26..24 are exactly zero; rank is 25 at NUM_FREQ=9 and only reaches 26 at
NUM_FREQ=10). Because lambda is a per-frame scalar, the reachable set of
per-frame key-logit vectors was therefore a fixed 23-dimensional subspace of
R^26 — three emission directions, concentrated on ``d/f/g/h/j/k`` and
``e/r/t/y/u/i``, were structurally unreachable no matter the width, depth, epoch
count or data volume.

The fix keeps the exported ``coefficients`` head at 64 wide (the phase-2
refinement head consumes ``concat(sliced[27], coeff[64], lambda[1]) = [T',92]``,
so that width is contract-frozen) and instead makes the *key side* learned:
each key is embedded from its own geometry — ``(cx, cy)`` plus the same 64
cosine features, which give the MLP a smooth multi-scale positional encoding —
through ``Linear(66,96) -> GELU -> Linear(96,64)``. Scoring is then a plain
batched matmul, ``key_logits = coeff @ keyEmbed^T * lambda``, which is both
full-rank-capable and ONNX ``MatMul`` (audit fix #9: no ``Einsum`` node).

Layout-agnosticism is preserved: the key embedding is a function of key geometry
only, never of slot index, so the slot-permutation augmentation still applies.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_KEYS = 64
T_IN = 64
T_OUT = 32          # one stride-2 stem: 64 -> 32
NUM_FREQ = 8        # 8x8 cosine features -> 64-dim positional encoding per key
NUM_COEFF = NUM_FREQ * NUM_FREQ   # 64 — contract-frozen `coefficients` width
MASK_NEG = -1.0e4   # finite "off" logit for pad key slots
EMBED_HID = 96      # hidden width of the per-key embedding MLP


def path_features(features: torch.Tensor) -> torch.Tensor:
    """``[B,2,64]`` raw path -> ``[B,8,64]`` derived channels.

    Computed in-graph so the exported ONNX consumes exactly the raw ``[2,64]``
    tensor ``CtcFeaturizer.featurize`` produces.
    """
    x = features[:, 0, :]
    y = features[:, 1, :]
    dx = F.pad(x[:, 1:] - x[:, :-1], (1, 0))
    dy = F.pad(y[:, 1:] - y[:, :-1], (1, 0))
    speed = torch.sqrt(dx * dx + dy * dy + 1e-8)
    ang = torch.atan2(dy, dx + 1e-8)
    arc = torch.cumsum(speed, dim=1)                    # cumulative arc length
    return torch.stack(
        [x, y, dx, dy, speed, torch.sin(ang), torch.cos(ang), arc], dim=1)


class ResBlock(nn.Module):
    """Dilated temporal-conv residual block (the TCN body)."""

    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        pad = 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class CtcSwipeEncoder(nn.Module):
    """TCN encoder + learned key-embedding scoring head.

    :param ch: TCN channel width.
    :param dilations: dilation per residual block.
    :param embed_hid: hidden width of the per-key embedding MLP.
    """

    def __init__(self, ch: int = 96, dilations: tuple = (1, 2, 4, 8),
                 embed_hid: int = EMBED_HID) -> None:
        super().__init__()
        self.ch = ch
        self.embed_hid = embed_hid
        self.stem = nn.Conv1d(8, ch, 5, stride=2, padding=2)      # 64 -> 32 frames
        self.stem_norm = nn.GroupNorm(8, ch)
        self.blocks = nn.ModuleList(ResBlock(ch, d) for d in dilations)
        self.coeff_head = nn.Linear(ch, NUM_COEFF)                # [B,32,64]
        self.lambda_head = nn.Linear(ch, 1)                       # positive gate
        self.blank_head = nn.Linear(ch, 1)                        # CTC blank logit
        # Per-key embedding: (cx, cy) + 64 cosine positional features -> 64-dim.
        self.key_embed = nn.Sequential(
            nn.Linear(2 + NUM_COEFF, embed_hid),
            nn.GELU(),
            nn.Linear(embed_hid, NUM_COEFF),
        )
        self.register_buffer("freq", torch.arange(NUM_FREQ, dtype=torch.float32))

    def key_positional(self, layout_keys: torch.Tensor) -> torch.Tensor:
        """``[B,64,2]`` key centers -> ``[B,64,66]`` geometry features.

        The 64 cosine terms ``cos(pi*u*cx) * cos(pi*v*cy)`` are a smooth
        multi-scale positional encoding; the raw ``(cx, cy)`` pair is appended so
        the MLP always has an exact, unaliased handle on position.
        """
        cx = layout_keys[..., 0]                                  # [B,64]
        cy = layout_keys[..., 1]
        bx = torch.cos(math.pi * self.freq[None, None, :] * cx[:, :, None])  # [B,64,8]
        by = torch.cos(math.pi * self.freq[None, None, :] * cy[:, :, None])  # [B,64,8]
        basis = (bx[:, :, :, None] * by[:, :, None, :]).reshape(
            cx.shape[0], MAX_KEYS, NUM_COEFF)                     # [B,64,64]
        return torch.cat([layout_keys, basis], dim=-1)            # [B,64,66]

    def forward(self, features: torch.Tensor, layout_keys: torch.Tensor,
                layout_mask: torch.Tensor):
        h = F.gelu(self.stem_norm(self.stem(path_features(features))))  # [B,ch,32]
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)                                     # [B,32,ch]

        coeff = self.coeff_head(h)                                # [B,32,64]
        lam = F.softplus(self.lambda_head(h))                     # [B,32,1] > 0
        blank = self.blank_head(h)                                # [B,32,1]

        # Learned, full-rank-capable key scoring (audit fix #2/#9: MatMul, no Einsum).
        key_vec = self.key_embed(self.key_positional(layout_keys))    # [B,64,64]
        key_logits = torch.matmul(coeff, key_vec.transpose(1, 2)) * lam  # [B,32,64]
        key_logits = torch.where(layout_mask[:, None, :], key_logits,
                                 torch.full_like(key_logits, MASK_NEG))

        logits = torch.cat([key_logits, blank], dim=-1)           # [B,32,65]
        log_emissions = F.log_softmax(logits, dim=-1)
        return log_emissions, coeff, lam

    @torch.no_grad()
    def key_embedding_matrix(self, centers: torch.Tensor) -> torch.Tensor:
        """``[K,2]`` key centers -> ``[K,64]`` embedding rows (rank-check helper)."""
        padded = torch.zeros(1, MAX_KEYS, 2, dtype=torch.float32,
                             device=centers.device)
        k = centers.shape[0]
        padded[0, :k] = centers
        return self.key_embed(self.key_positional(padded))[0, :k]


# ── phase 2: the refinement head (guide §11) ────────────────────────────────────

def slice_head_torch(log_emissions: torch.Tensor, num_letters: int) -> torch.Tensor:
    """``[B,T,MAX_KEYS+1]`` full head -> ``[B,T,num_letters+1]`` contract view.

    The torch twin of ``futo_decoder_ceiling.slice_emissions`` and of Kotlin's
    ``CtcEmissions.sliceFromHead``: keep the K real key columns and relocate the
    blank from full-head column ``MAX_KEYS`` to column ``num_letters``.

    Valid only for the canonical *identity* slot assignment (key ``c`` in slot
    ``c``); under the slot-permutation augmentation the first ``num_letters``
    columns are not the alphabet, which is why the refinement head is trained
    without permutation and is canonical-QWERTY-gated.
    """
    return torch.cat([log_emissions[..., :num_letters],
                      log_emissions[..., MAX_KEYS:MAX_KEYS + 1]], dim=-1)


def refine_input_dim(num_letters: int) -> int:
    """Per-frame width of ``concat(sliced | coefficients | lambda)`` (92 for a-z)."""
    return (num_letters + 1) + NUM_COEFF + 1


class CtcRefineHead(nn.Module):
    """Per-frame refinement head — our ``magic_macaw`` analogue (guide §11).

    Consumes ``concat(sliced_emissions[K+1], coefficients[64], lambda[1])`` and
    emits refined ``log_probs[K+1]`` that REPLACE the emissions before the beam.
    Because it sees the spatial coefficients alongside the collapsed emissions it
    can sharpen per-frame decisions using context the key softmax discarded —
    FUTO measured +5.88 pt top-1 (greedy 43.96 % -> 69.12 %) from this lever.

    Blank sits at index ``num_letters`` here (26), not 64: the head operates on
    the already-sliced view.

    :param num_letters: active alphabet size (26 for en_qwerty).
    :param hidden: hidden width of the MLP.
    """

    def __init__(self, num_letters: int = 26, hidden: int = 128) -> None:
        super().__init__()
        self.num_letters = num_letters
        self.hidden = hidden
        self.in_dim = refine_input_dim(num_letters)
        self.norm = nn.LayerNorm(self.in_dim)
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_letters + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,T,in_dim]`` -> log-softmaxed ``[B,T,num_letters+1]``."""
        h = F.gelu(self.fc1(self.norm(x)))
        return F.log_softmax(self.fc2(h), dim=-1)


def build_refine_input(log_emissions: torch.Tensor, coeff: torch.Tensor,
                       lam: torch.Tensor, num_letters: int) -> torch.Tensor:
    """Assemble the head's ``[B,T,92]`` input from the base encoder's three heads.

    Mirrors ``futo_decoder_ceiling.build_decoder_input`` exactly.
    """
    sliced = slice_head_torch(log_emissions, num_letters)
    return torch.cat([sliced, coeff, lam], dim=-1)
