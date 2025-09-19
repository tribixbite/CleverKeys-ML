#!/usr/bin/env python3
"""
Simple demonstration of how rare_words sampling profile affects word selection.
"""

import numpy as np
from collections import Counter
from sampling_profiles import get_profile

# Example word pool with frequencies
WORD_POOL = [
    # Very common words
    ("the", 50000), ("and", 40000), ("you", 30000), ("that", 25000), ("this", 20000),

    # Common words
    ("hello", 5000), ("world", 4000), ("today", 3000), ("phone", 2500), ("time", 2000),

    # Medium frequency
    ("keyboard", 500), ("gesture", 400), ("swipe", 300), ("typing", 250), ("mobile", 200),

    # Rare words (≤50 occurrences - these get 5x boost)
    ("cryptocurrency", 45), ("blockchain", 40), ("quantum", 35), ("neural", 30), ("algorithm", 25),

    # Very rare words
    ("anthropomorphic", 10), ("serendipity", 8), ("onomatopoeia", 5), ("syzygy", 2), ("quixotic", 1),

    # Technical/proper nouns
    ("kubernetes", 15), ("tensorflow", 20), ("pytorch", 25), ("jupyter", 12), ("anthropic", 3),
]

def compute_weight_for_profile(word: str, freq: int, profile: dict) -> float:
    """Calculate sampling weight for a word using the profile settings."""

    # Skip if word is too short
    min_length = profile.get("min_word_length", 1)
    if len(word) < min_length:
        return 0.0

    # Apply inverse frequency weighting
    freq_power = profile.get("freq_power", 0.5)
    weight = freq ** (-abs(freq_power))

    # Apply rare word boost (5x for words with ≤50 occurrences)
    rare_threshold = profile.get("rare_frequency_threshold", 0)
    rare_boost = profile.get("rare_word_boost", 1.0)
    if rare_threshold and freq <= rare_threshold:
        weight *= rare_boost

    return weight

def simulate_sampling(profile_name: str, num_samples: int = 100):
    """Simulate sampling with a given profile."""

    profile = get_profile(profile_name)

    # Calculate weights for all words
    weights = []
    words = []
    for word, freq in WORD_POOL:
        w = compute_weight_for_profile(word, freq, profile)
        if w > 0:
            weights.append(w)
            words.append((word, freq))

    # Normalize to probabilities
    weights = np.array(weights)
    if profile_name == "base_random":
        # Equal probability for all words
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # Sample words
    np.random.seed(42)  # Reproducible
    sampled_indices = np.random.choice(len(words), size=num_samples, p=weights)
    sampled_words = [words[i][0] for i in sampled_indices]

    return Counter(sampled_words), words, weights

def main():
    print("=" * 80)
    print("RARE WORDS SAMPLING PROFILE DEMONSTRATION")
    print("=" * 80)
    print()

    print("## How the 'rare_words' profile works:\n")
    print("1. Inverse Frequency Weighting (freq^-0.7):")
    print("   - Common word (freq=50,000): weight ∝ 1/(50000^0.7) = 0.00041")
    print("   - Rare word (freq=10): weight ∝ 1/(10^0.7) = 0.1995")
    print("   - Ratio: Rare word is ~487x more likely to be sampled!\n")

    print("2. Rare Word Boost (5x for words with ≤50 occurrences):")
    print("   - Words appearing ≤50 times get additional 5x multiplier")
    print("   - Combined effect: rare words can be 2000x+ more likely than common words\n")

    print("3. Minimum Word Length: 3 characters")
    print("   - Filters out 'a', 'an', 'to', etc.\n")

    print("-" * 80)

    # Compare profiles
    profiles = ["base_random", "rare_words"]

    for profile_name in profiles:
        print(f"\n## Profile: {profile_name}")
        profile = get_profile(profile_name)
        print(f"Description: {profile.get('description', 'N/A')}\n")

        # Simulate sampling
        sampled_counts, word_list, weights = simulate_sampling(profile_name, 1000)

        # Show sampling probabilities for different frequency ranges
        print("Sampling probabilities by frequency range:")

        ranges = [
            ("Very Common (>10,000)", lambda f: f > 10000),
            ("Common (1,000-10,000)", lambda f: 1000 < f <= 10000),
            ("Medium (100-1,000)", lambda f: 100 < f <= 1000),
            ("Rare (≤50)", lambda f: f <= 50),
            ("Very Rare (≤10)", lambda f: f <= 10),
        ]

        for range_name, range_filter in ranges:
            # Find words in this range
            range_indices = [i for i, (w, f) in enumerate(word_list) if range_filter(f)]
            if range_indices:
                range_weights = weights[range_indices]
                total_prob = range_weights.sum()
                avg_prob = range_weights.mean()
                example = word_list[range_indices[0]][0]
                print(f"  {range_name:25s}: {total_prob*100:5.1f}% total, {avg_prob*100:5.2f}% avg per word (e.g., '{example}')")

        print(f"\nTop 10 most sampled words (out of 1000 samples):")
        for word, count in sampled_counts.most_common(10):
            freq = next(f for w, f in WORD_POOL if w == word)
            pct = count / 10  # Percentage
            print(f"  - {word:20s} ({count:3d} times = {pct:4.1f}%) [actual freq: {freq:,}]")

        # Count how many unique rare vs common words were sampled
        rare_sampled = set(w for w in sampled_counts if any(w == word and f <= 50 for word, f in WORD_POOL))
        common_sampled = set(w for w in sampled_counts if any(w == word and f > 1000 for word, f in WORD_POOL))

        print(f"\nDiversity in 1000 samples:")
        print(f"  - Unique rare words (≤50 freq) sampled: {len(rare_sampled)}/{sum(1 for _, f in WORD_POOL if f <= 50)}")
        print(f"  - Unique common words (>1000 freq) sampled: {len(common_sampled)}/{sum(1 for _, f in WORD_POOL if f > 1000)}")

    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("With rare_words profile, the model sees uncommon words MUCH more frequently,")
    print("helping it learn proper nouns, technical terms, and rare vocabulary that would")
    print("otherwise be underrepresented in training.")
    print("=" * 80)

if __name__ == "__main__":
    main()