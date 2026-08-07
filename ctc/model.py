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

#: v1 path channels; v2 adds d2x, d2y, curvature, progress and two edge ramps.
FEAT_V1 = 8
FEAT_V2_PATH = 14
#: Width of the learned reduction applied to the [T, MAX_KEYS] proximity field.
PROX_PROJ = 16
#: Ramp length (frames) for the is_start / is_end channels.
RAMP = 8
#: Half the diagonal of a canonical en_qwerty key (width 1/10, height 1/3):
#: 0.5 * sqrt(0.1^2 + (1/3)^2). Only sets the *initial* proximity scale — both
#: proximity parameters are learned, so a different layout is not baked in.
HALF_KEY_DIAG = 0.5 * math.sqrt(0.1 ** 2 + (1.0 / 3.0) ** 2)


def feat_channels(feat_version: int) -> int:
    """Trunk input width for a feature version."""
    return FEAT_V1 if feat_version == 1 else FEAT_V2_PATH + PROX_PROJ


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


def path_features_v2(features: torch.Tensor) -> torch.Tensor:
    """``[B,2,64]`` raw path -> ``[B,14,64]``: the v1 eight plus six kinematics.

    Added channels, all differentiable and all ONNX-elementwise:

    * ``d2x``/``d2y`` — second differences, so the trunk sees acceleration
      without having to synthesise it through two conv layers;
    * ``curvature`` — ``(dx*d2y - dy*d2x) / (dx^2 + dy^2)^{3/2}``, the signed
      turn rate that distinguishes a cusp at a key from a smooth pass-through.
      Clamped because the denominator vanishes wherever the resampler produced a
      stationary frame;
    * ``progress`` — ``arc / arc_total``, the scale-free position along the
      stroke. ``arc`` alone is not comparable between a short and a long word;
    * ``is_start``/``is_end`` — linear ramps over the first and last
      :data:`RAMP` frames. The featurizer resamples every trace to exactly 64
      frames, so these are positional constants that tell the causal-free conv
      where the boundaries are.
    """
    x = features[:, 0, :]
    y = features[:, 1, :]
    dx = F.pad(x[:, 1:] - x[:, :-1], (1, 0))
    dy = F.pad(y[:, 1:] - y[:, :-1], (1, 0))
    speed = torch.sqrt(dx * dx + dy * dy + 1e-8)
    ang = torch.atan2(dy, dx + 1e-8)
    arc = torch.cumsum(speed, dim=1)

    d2x = F.pad(dx[:, 1:] - dx[:, :-1], (1, 0))
    d2y = F.pad(dy[:, 1:] - dy[:, :-1], (1, 0))
    denom = (dx * dx + dy * dy + 1e-6) ** 1.5
    curv = torch.clamp((dx * d2y - dy * d2x) / denom, -50.0, 50.0)
    progress = arc / (arc[:, -1:] + 1e-6)

    t = torch.arange(features.shape[-1], dtype=features.dtype,
                     device=features.device)
    n = features.shape[-1] - 1
    start = torch.clamp(1.0 - t / RAMP, min=0.0).expand_as(x)
    end = torch.clamp(1.0 - (n - t) / RAMP, min=0.0).expand_as(x)

    return torch.stack([x, y, dx, dy, speed, torch.sin(ang), torch.cos(ang),
                        arc, d2x, d2y, curv, progress, start, end], dim=1)


class KeyProximity(nn.Module):
    """Soft path-to-key proximity field, reduced to :data:`PROX_PROJ` channels.

    For every frame ``t`` and key slot ``k`` this scores how close the path is to
    that key, ``prox[t,k] = sigmoid(a - b * dist(path_t, key_k))``, zeroes the pad
    slots through ``layout_mask``, and projects the ``MAX_KEYS``-wide vector down
    with a learned ``Linear``. The trunk therefore receives an explicit "which
    keys am I over right now" signal instead of having to rediscover the layout
    geometry from raw coordinates every forward pass.

    The functional form is proshian's measured one, ``sigmoid(-1.8*d/half_key_diag
    + 4)``; ``a`` and ``b`` are initialised to that and then **learned**, so the
    layout-agnosticism the slot-permutation augmentation depends on is preserved —
    nothing here is a function of slot index, only of geometry.

    ONNX: broadcast sub/mul, Sqrt, Sigmoid, Cast and MatMul only — no Einsum.
    """

    def __init__(self, out_ch: int = PROX_PROJ) -> None:
        super().__init__()
        self.proj = nn.Linear(MAX_KEYS, out_ch)
        self.a = nn.Parameter(torch.tensor(4.0))
        self.b = nn.Parameter(torch.tensor(1.8 / HALF_KEY_DIAG))

    def forward(self, features: torch.Tensor, layout_keys: torch.Tensor,
                layout_mask: torch.Tensor) -> torch.Tensor:
        """-> ``[B, out_ch, T]`` proximity summary."""
        px = features[:, 0, :].unsqueeze(-1)                  # [B,T,1]
        py = features[:, 1, :].unsqueeze(-1)
        kx = layout_keys[..., 0].unsqueeze(1)                 # [B,1,K]
        ky = layout_keys[..., 1].unsqueeze(1)
        ddx = px - kx
        ddy = py - ky
        dist = torch.sqrt(ddx * ddx + ddy * ddy + 1e-8)       # [B,T,K]
        prox = torch.sigmoid(self.a - self.b * dist)
        prox = prox * layout_mask.unsqueeze(1).to(prox.dtype)
        return self.proj(prox).transpose(1, 2)                # [B,out_ch,T]


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


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2), over the temporal axis.

    Operates channels-last on ``[B,T,C]``: the per-channel L2 energy across time
    is divided by its mean over channels, which re-weights channels against each
    other and counteracts the feature collapse a plain residual MLP tends to.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(ch))
        self.beta = nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-12)   # [B,1,C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class SEGate(nn.Module):
    """Squeeze-and-excitation over the temporal axis, channels-first ``[B,C,T]``."""

    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        hid = max(4, ch // reduction)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)                                  # [B,C]
        s = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))
        return x * s.unsqueeze(-1)


class ConvNeXtBlock(nn.Module):
    """Dilated depthwise conv -> LN -> 1x1 expand + GLU -> GRN -> project -> SE.

    The FUTO-paper trunk shape. ``expand=4`` means the pointwise layer widens to
    ``4*ch`` and the GLU halves it again to ``2*ch``, which is the inner width the
    GRN and the projection see; that reading is what keeps the parameter count in
    the intended band, and it is reported explicitly rather than assumed.

    Normalization is LayerNorm (channels-last), never BatchNorm — inference runs
    at batch 1 on-device and batch statistics would be meaningless there.
    """

    def __init__(self, ch: int, dilation: int, kernel: int = 5,
                 expand: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, kernel, padding=(kernel // 2) * dilation,
                            dilation=dilation, groups=ch)
        self.norm = nn.LayerNorm(ch)
        inner = ch * expand
        self.pw1 = nn.Linear(ch, inner)          # GLU halves inner -> ch*expand/2
        self.grn = GRN(inner // 2)
        self.pw2 = nn.Linear(inner // 2, ch)
        self.se = SEGate(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)                                       # [B,C,T]
        h = h.transpose(1, 2)                                # [B,T,C]
        h = self.norm(h)
        h = F.glu(self.pw1(h), dim=-1)
        h = self.grn(h)
        h = self.pw2(h).transpose(1, 2)                      # [B,C,T]
        return x + self.se(h)


class CtcSwipeEncoder(nn.Module):
    """TCN encoder + learned key-embedding scoring head.

    :param ch: TCN channel width.
    :param dilations: dilation per residual block.
    :param embed_hid: hidden width of the per-key embedding MLP.
    :param feat_version: 1 = the original 8 path channels; 2 = 14 kinematic
        channels plus a learned reduction of the key-proximity field.
    :param block: ``"res"`` for the original two-conv residual block, ``"convnext"``
        for the depthwise/GLU/GRN/SE block.
    """

    def __init__(self, ch: int = 96, dilations: tuple = (1, 2, 4, 8),
                 embed_hid: int = EMBED_HID, feat_version: int = 1,
                 block: str = "res") -> None:
        super().__init__()
        self.ch = ch
        self.embed_hid = embed_hid
        self.feat_version = feat_version
        self.block = block
        self.dilations = tuple(dilations)
        in_ch = feat_channels(feat_version)
        self.prox = KeyProximity() if feat_version >= 2 else None
        self.stem = nn.Conv1d(in_ch, ch, 5, stride=2, padding=2)  # 64 -> 32 frames
        self.stem_norm = nn.GroupNorm(8, ch)
        if block == "convnext":
            self.blocks = nn.ModuleList(ConvNeXtBlock(ch, d) for d in dilations)
        else:
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

    def trunk_input(self, features: torch.Tensor, layout_keys: torch.Tensor,
                    layout_mask: torch.Tensor) -> torch.Tensor:
        """Assemble the stem's input channels for this feature version."""
        if self.feat_version >= 2:
            return torch.cat([path_features_v2(features),
                              self.prox(features, layout_keys, layout_mask)], dim=1)
        return path_features(features)

    def forward(self, features: torch.Tensor, layout_keys: torch.Tensor,
                layout_mask: torch.Tensor):
        h = self.trunk_input(features, layout_keys, layout_mask)
        h = F.gelu(self.stem_norm(self.stem(h)))                        # [B,ch,32]
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


def encoder_from_checkpoint(ck: dict) -> CtcSwipeEncoder:
    """Rebuild the exact architecture a checkpoint was trained with.

    Every consumer (export, eval, latency probe) goes through this so a new arch
    knob cannot be honoured in training and silently ignored at export time.
    Defaults reproduce the pre-Phase-B architecture for old checkpoints.
    """
    return CtcSwipeEncoder(
        ch=ck.get("ch", 96),
        embed_hid=ck.get("embed_hid", EMBED_HID),
        dilations=tuple(ck.get("dilations", (1, 2, 4, 8))),
        feat_version=int(ck.get("feat_version", 1)),
        block=str(ck.get("block", "res")),
    )


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
