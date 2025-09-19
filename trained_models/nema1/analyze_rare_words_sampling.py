#!/usr/bin/env python3
"""
Analyze and demonstrate how the rare_words sampling profile affects word selection.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Import the sampling profiles
from sampling_profiles import get_profile

def load_word_frequencies() -> Dict[str, int]:
    """Load word frequencies from training data."""
    train_file = Path("../../data/train_final_train.jsonl")
    word_counts = Counter()

    if train_file.exists():
        with open(train_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                word = sample["word"]
                word_counts[word] += 1
    return dict(word_counts)

def compute_weight(word: str, freq: int, profile: Dict) -> float:
    """Compute sampling weight for a word given the profile."""
    freq_power = profile.get("freq_power", 0.5)
    length_power = profile.get("length_power", 0.0)
    rare_threshold = profile.get("rare_frequency_threshold", 0)
    rare_boost = profile.get("rare_word_boost", 1.0)
    min_length = profile.get("min_word_length", 1)
    max_length = profile.get("max_word_length")
    max_freq = profile.get("max_frequency")

    word_len = len(word)

    # Filter by length
    if word_len < min_length:
        return 0.0
    if max_length and word_len > max_length:
        return 0.0
    if max_freq and freq > max_freq:
        return 0.0

    # Base weight from inverse frequency
    weight = freq ** (-abs(freq_power))

    # Apply length power
    if length_power:
        weight *= word_len ** length_power

    # Apply rare word boost
    if rare_threshold and freq <= rare_threshold:
        weight *= rare_boost

    return weight

def analyze_profile(profile_name: str):
    """Analyze how a profile affects sampling."""
    profile = get_profile(profile_name)
    word_freqs = load_word_frequencies()

    if not word_freqs:
        print("Could not load word frequencies. Using synthetic data...")
        # Create synthetic data for demonstration
        word_freqs = {
            # Very common words
            "the": 50000, "and": 40000, "you": 30000, "that": 25000,
            # Common words
            "hello": 5000, "world": 4000, "today": 3000, "phone": 2500,
            # Medium frequency
            "keyboard": 500, "gesture": 400, "swipe": 300, "typing": 250,
            # Rare words (< 50 occurrences)
            "cryptocurrency": 45, "blockchain": 40, "quantum": 35, "neural": 30,
            # Very rare words
            "anthropomorphic": 10, "serendipity": 8, "onomatopoeia": 5, "syzygy": 2,
            # Proper nouns / specialized
            "kubernetes": 15, "tensorflow": 20, "pytorch": 25, "nemo": 12,
        }

    # Calculate weights for all words
    word_weights = []
    for word, freq in word_freqs.items():
        weight = compute_weight(word, freq, profile)
        if weight > 0:
            word_weights.append((word, freq, weight))

    # Sort by weight
    word_weights.sort(key=lambda x: x[2], reverse=True)

    # Normalize weights
    total_weight = sum(w[2] for w in word_weights)
    word_weights = [(w, f, wt/total_weight * len(word_weights)) for w, f, wt in word_weights]

    return word_weights

def generate_comparison_report():
    """Generate a report comparing different profiles."""

    profiles_to_compare = ["base_random", "production_current", "rare_words", "very_rare"]

    print("=" * 80)
    print("RARE WORDS SAMPLING PROFILE ANALYSIS")
    print("=" * 80)
    print()

    print("## What the rare_words profile does:\n")
    print("The 'rare_words' sampling profile dramatically changes how training data is selected:")
    print("1. **Inverse Frequency Weighting (freq^-0.7)**: Words get weight = 1/freq^0.7")
    print("   - Common word (freq=1000): weight = 0.063")
    print("   - Rare word (freq=10): weight = 0.398 (6.3x more likely)")
    print()
    print("2. **Rare Word Boost (5x)**: Words with ≤50 occurrences get 5x additional boost")
    print("   - Word with freq=45: base weight * 5 = massive oversampling")
    print()
    print("3. **Effect**: The model sees rare words MUCH more often during training")
    print("   - Helps learn uncommon vocabulary, proper nouns, technical terms")
    print("   - Reduces bias toward common words like 'the', 'and', 'you'")
    print()
    print("-" * 80)

    for profile_name in profiles_to_compare:
        print(f"\n## Profile: {profile_name}")
        profile_info = get_profile(profile_name)
        print(f"Description: {profile_info.get('description', 'N/A')}")
        print()

        word_weights = analyze_profile(profile_name)

        # Show top 10 highest weighted words
        print("Top 10 highest weighted words (relative probability):")
        for i, (word, freq, weight) in enumerate(word_weights[:10], 1):
            print(f"  {i:2d}. {word:20s} (freq={freq:6d}) -> weight={weight:6.3f}x")

        # Show some common words for comparison
        print("\nCommon word weights:")
        common_words = ["the", "and", "you", "hello", "world"]
        for word in common_words:
            for w, f, wt in word_weights:
                if w == word:
                    print(f"  - {word:20s} (freq={f:6d}) -> weight={wt:6.3f}x")
                    break

        # Calculate statistics
        rare_words = [w for w in word_weights if w[1] <= 50]
        common_words = [w for w in word_weights if w[1] > 1000]

        if rare_words and common_words:
            rare_avg_weight = np.mean([w[2] for w in rare_words])
            common_avg_weight = np.mean([w[2] for w in common_words])

            print(f"\nStatistics:")
            print(f"  - Avg weight for rare words (≤50 freq): {rare_avg_weight:.3f}x")
            print(f"  - Avg weight for common words (>1000 freq): {common_avg_weight:.3f}x")
            print(f"  - Rare/Common ratio: {rare_avg_weight/max(common_avg_weight, 0.001):.1f}x")

def generate_sampling_examples():
    """Generate example training batches to show the difference."""

    print("\n" + "=" * 80)
    print("EXAMPLE TRAINING BATCHES")
    print("=" * 80)
    print()

    # Simulate word pool
    word_pool = [
        ("the", 50000), ("and", 40000), ("you", 30000), ("hello", 5000),
        ("world", 4000), ("phone", 2500), ("keyboard", 500), ("gesture", 400),
        ("cryptocurrency", 45), ("blockchain", 40), ("quantum", 35),
        ("anthropomorphic", 10), ("serendipity", 8), ("kubernetes", 15),
    ]

    for profile_name in ["base_random", "rare_words"]:
        print(f"\n## Simulated batch with {profile_name} profile:")
        profile = get_profile(profile_name)

        # Calculate weights
        weights = []
        words = []
        for word, freq in word_pool:
            w = compute_weight(word, freq, profile)
            if w > 0:
                weights.append(w)
                words.append(word)

        # Normalize to probabilities
        weights = np.array(weights)
        if profile_name == "base_random":
            # Equal probability for all
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        # Sample 20 words
        np.random.seed(42)  # For reproducibility
        sampled_indices = np.random.choice(len(words), size=20, p=weights)
        sampled_words = [words[i] for i in sampled_indices]

        # Count occurrences
        word_counts = Counter(sampled_words)

        print("Sampled words in batch of 20:")
        for word, count in word_counts.most_common():
            # Find frequency for this word
            word_freq = 0
            for w, f in word_pool:
                if w == word:
                    word_freq = f
                    break
            print(f"  - {word:20s} appears {count:2d} times (freq in data: {word_freq:6d})")

if __name__ == "__main__":
    generate_comparison_report()
    generate_sampling_examples()

    # Save detailed analysis to file
    with open("rare_words_analysis.txt", "w") as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        generate_comparison_report()
        generate_sampling_examples()
        sys.stdout = old_stdout

    print("\n" + "=" * 80)
    print("Analysis saved to rare_words_analysis.txt")
    print("=" * 80)