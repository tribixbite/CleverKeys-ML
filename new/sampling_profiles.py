#!/usr/bin/env python3
"""
Sampling Profiles for CleverKeys RNNT Training

This file contains different sampling strategies for training the swipe gesture model.
Each profile targets different word characteristics to improve model performance on
specific types of words.

The sampling system uses weighted random sampling to adjust the frequency of training
examples based on word properties like frequency, length, and rarity.
"""

SAMPLING_PROFILES = {
    # 1. BASE RANDOM - Uniform sampling (original behavior before sampling config)
    "base_random": {
        "strategy": "none",  # Disables weighted sampling
        "description": "Uniform random sampling - baseline for comparing with legacy checkpoints"
    },

    # 2. RARE WORDS - Focus on low-frequency words
    "rare_words": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.7,  # Strong inverse frequency weighting
        "rare_frequency_threshold": 50,  # Words appearing ≤50 times
        "rare_word_boost": 5.0,  # 5x boost for rare words
        "max_weight_factor": 15.0,
        "min_word_length": 3,  # Skip very short words
        "description": "Heavily samples rare and uncommon words to improve tail distribution"
    },

    # 3. LONG WORDS - Focus on longer words (8+ characters)
    "long_words": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.3,  # Moderate frequency weighting
        "length_power": 1.5,  # Strong length weighting
        "min_word_length": 8,  # Only words 8+ chars
        "max_weight_factor": 20.0,
        "description": "Targets longer words which are harder to swipe accurately"
    },

    # 4. SHORT COMMON - Focus on short common words (3-5 chars, high frequency)
    "short_common": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": -0.3,  # POSITIVE weight for common words
        "min_word_length": 3,
        "max_word_length": 5,
        "max_frequency": 10000,  # Only common words
        "max_weight_factor": 8.0,
        "description": "Short frequent words that are critical for fluent typing"
    },

    # 5. MEDIUM BALANCED - Medium length words (5-8 chars) with balanced frequency
    "medium_balanced": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.4,
        "length_power": 0.3,
        "min_word_length": 5,
        "max_word_length": 8,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 2.0,
        "max_weight_factor": 10.0,
        "description": "Balanced sampling for medium-length words"
    },

    # 6. VERY RARE - Ultra-rare words and proper nouns
    "very_rare": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.9,  # Very strong inverse frequency
        "rare_frequency_threshold": 10,  # Words appearing ≤10 times
        "rare_word_boost": 10.0,  # 10x boost
        "min_word_length": 4,
        "max_weight_factor": 30.0,
        "description": "Extreme focus on very rare words, proper nouns, specialized terms"
    },

    # 7. HIGH CONFUSION - Words that often get confused (similar swipe paths)
    "high_confusion": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.6,
        "min_word_length": 4,
        "max_word_length": 7,  # Mid-length words have most confusion
        "rare_frequency_threshold": 200,
        "rare_word_boost": 3.0,
        "max_weight_factor": 12.0,
        "description": "Words prone to confusion due to similar swipe patterns"
    },

    # 8. PRODUCTION (Current Settings) - What's currently in use
    "production_current": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.55,
        "length_power": 0.8,
        "rare_frequency_threshold": 25,
        "rare_word_boost": 3.5,
        "max_weight_factor": 12.0,
        "min_word_length": 4,
        "max_frequency": 3000,
        "description": "Current production settings from train_transducer_personalized.py"
    },

    # VALIDATION PROFILES
    "validation_current": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.75,
        "length_power": 1.3,
        "min_word_length": 7,
        "rare_frequency_threshold": 40,
        "rare_word_boost": 6.0,
        "max_weight_factor": 28.0,
        "batch_size_factor": 0.35,
        "limit_batches": 0.15,
        "max_samples": 1500,
        "description": "Current validation settings - harder subset for robust evaluation"
    },

    "validation_balanced": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.5,
        "min_word_length": 3,
        "max_weight_factor": 5.0,
        "limit_batches": 0.3,
        "max_samples": 3000,
        "description": "More balanced validation for comparing across checkpoints"
    }
}

# Analysis of what to prioritize given current greedy decoding output
PRIORITY_RECOMMENDATIONS = """
Based on the current system using greedy decoding (which will be replaced by beam search):

1. **HIGHEST PRIORITY - Rare Words Profile**:
   - Greedy decoding particularly struggles with rare words
   - Beam search will help, but training on more rare examples is critical

2. **HIGH PRIORITY - Long Words Profile**:
   - Longer swipes accumulate more uncertainty
   - Greedy can't recover from early mistakes in long words

3. **MEDIUM PRIORITY - High Confusion Profile**:
   - Words with similar swipe patterns need more training examples
   - Beam search will explore alternatives, but model needs better discrimination

4. **LOWER PRIORITY - Short Common**:
   - These likely already work well
   - But important for user experience since they're typed frequently

For comparing with older checkpoints before sampling config:
- Use "base_random" profile to match original training distribution
- This allows fair WER comparison across checkpoint epochs

Current production settings seem well-tuned but could benefit from:
- Slightly higher freq_power (0.6-0.65) to focus more on rare words
- Consider separate models for common vs rare word specialists
"""


def get_profile(name: str) -> dict:
    """Get a sampling profile by name."""
    if name not in SAMPLING_PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(SAMPLING_PROFILES.keys())}")
    return SAMPLING_PROFILES[name].copy()


def print_profile_comparison():
    """Print a comparison table of all profiles."""
    print("\nSampling Profile Comparison")
    print("=" * 100)

    params = ["strategy", "freq_power", "length_power", "min_word_length",
              "max_word_length", "rare_frequency_threshold", "rare_word_boost",
              "max_weight_factor", "max_frequency"]

    print(f"{'Profile':<20}", end="")
    for param in params:
        print(f"{param[:8]:<9}", end="")
    print()
    print("-" * 100)

    for name, profile in SAMPLING_PROFILES.items():
        if "validation" in name:
            continue  # Skip validation profiles in main comparison
        print(f"{name:<20}", end="")
        for param in params:
            value = profile.get(param, "-")
            if isinstance(value, float):
                print(f"{value:<9.2f}", end="")
            elif value is None:
                print(f"{'None':<9}", end="")
            else:
                print(f"{str(value):<9}", end="")
        print()


if __name__ == "__main__":
    print_profile_comparison()
    print("\n" + PRIORITY_RECOMMENDATIONS)