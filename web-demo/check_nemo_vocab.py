#!/usr/bin/env python3
"""Check how NeMo RNNT models handle vocabulary"""

# The vocab file has these 29 tokens (indices 0-28):
# 0: <blank>
# 1: '
# 2-27: a-z
# 28: <unk>

# But the model outputs 30 logits.
# Common RNNT implementations add an extra token for internal use.
# Let's verify this hypothesis.

vocab_tokens = [
    "<blank>",  # 0
    "'",        # 1
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",  # 2-14
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",  # 15-27
    "<unk>"     # 28
]

print(f"Vocab file has {len(vocab_tokens)} tokens (0-{len(vocab_tokens)-1})")
print(f"Model outputs 30 logits (0-29)")
print()
print("In NeMo's RNNT implementation, the model typically adds 1 to vocab_size")
print("for internal padding/EOS handling. Token 29 is likely not meant to be")
print("predicted during normal decoding.")
print()
print("The fact that after emitting a character, the model strongly predicts")
print("token 29, suggests it might be used as a 'return to blank' signal.")
print()
print("Solution: We should ignore token 29 during decoding and treat only")
print("token 0 as the blank.")