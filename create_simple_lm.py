#!/usr/bin/env python3
"""
Create a simple unigram ARPA language model from our vocabulary.
This is a basic model but better than no language model at all.
"""

import math
from collections import Counter

# Read vocabulary and assign frequencies
# In a real scenario, you'd want actual word frequencies from a corpus
print("Reading vocabulary...")
with open('vocab/final_vocab.txt', 'r') as f:
    words = [word.strip().lower() for word in f.readlines()]

# Create a simple frequency distribution (Zipf's law approximation)
# Most common words get higher probability
word_freq = {}
for i, word in enumerate(sorted(words, key=len)):  # Sort by length as proxy for frequency
    # Simple Zipf distribution
    word_freq[word] = 1.0 / (i + 1)

# Normalize to get probabilities
total = sum(word_freq.values())
word_prob = {w: freq/total for w, freq in word_freq.items()}

# Add OOV (out of vocabulary) probability
oov_prob = 1e-6
word_prob['<unk>'] = oov_prob

# Write ARPA format
print("Writing simple_lm.arpa...")
with open('simple_lm.arpa', 'w') as f:
    # Header
    f.write("\\data\\\n")
    f.write(f"ngram 1={len(word_prob)}\n")
    f.write("\n")
    
    # Unigrams
    f.write("\\1-grams:\n")
    
    # Write log probabilities
    for word, prob in word_prob.items():
        log_prob = math.log10(prob) if prob > 0 else -99
        f.write(f"{log_prob:.6f}\t{word}\n")
    
    # End
    f.write("\n\\end\\\n")

print(f"Created simple_lm.arpa with {len(word_prob)} words")