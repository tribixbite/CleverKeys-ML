#!/usr/bin/env python3
"""
Balanced Sampling Profiles for Production Use

These profiles provide a more conservative approach that won't
cause the model to hallucinate rare words inappropriately.
"""

BALANCED_PROFILES = {
    # 1. MILD RARE BOOST - Safe for production
    "mild_rare_boost": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.3,  # Much gentler than 0.7
        "rare_frequency_threshold": 50,
        "rare_word_boost": 2.0,  # Only 2x instead of 5x
        "max_weight_factor": 5.0,  # Cap at 5x instead of 15x
        "description": "Gentle boost for rare words without breaking common words"
    },

    # 2. VOCABULARY COVERAGE - Ensure all words seen
    "vocabulary_coverage": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.2,  # Very mild frequency weighting
        "min_occurrences": 1,  # Ensure every word is seen at least once
        "guarantee_coverage": True,  # Special flag to ensure coverage
        "max_weight_factor": 3.0,
        "description": "Ensures model sees every word at least occasionally"
    },

    # 3. STRATIFIED SAMPLING - Balanced across frequency bands
    "stratified_balanced": {
        "strategy": "stratified",
        "frequency_bands": [
            {"min": 0, "max": 50, "weight": 0.25},      # Rare
            {"min": 51, "max": 500, "weight": 0.25},    # Uncommon
            {"min": 501, "max": 5000, "weight": 0.25},  # Common
            {"min": 5001, "max": None, "weight": 0.25}, # Very common
        ],
        "description": "Equal representation across frequency bands"
    },

    # 4. FOCUS ON MISTAKES - Sample based on current error patterns
    "error_focused": {
        "strategy": "error_weighted",
        "base_power": 0.3,
        "error_multiplier": 3.0,  # 3x weight for words model gets wrong
        "requires_error_log": True,  # Needs error analysis from validation
        "description": "Focus on words the model currently struggles with"
    },

    # 5. LENGTH-AWARE BALANCED
    "length_balanced": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.25,
        "length_power": 0.15,  # Slight boost for longer words
        "length_bands": [
            {"min": 1, "max": 3, "weight": 0.8},   # Short words less weight
            {"min": 4, "max": 6, "weight": 1.0},   # Normal weight
            {"min": 7, "max": 9, "weight": 1.2},   # Slight boost
            {"min": 10, "max": None, "weight": 1.5}, # More boost for long
        ],
        "max_weight_factor": 4.0,
        "description": "Balanced sampling with slight preference for longer words"
    },

    # 6. PRODUCTION SAFE - Very conservative
    "production_safe": {
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.15,  # Very mild
        "rare_frequency_threshold": 25,
        "rare_word_boost": 1.5,  # Only 50% boost
        "max_weight_factor": 2.0,  # Never more than 2x any word
        "description": "Ultra-conservative for production deployment"
    },
}


def get_balanced_profile(name: str) -> dict:
    """Get a balanced sampling profile by name."""
    if name not in BALANCED_PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(BALANCED_PROFILES.keys())}")
    return BALANCED_PROFILES[name].copy()


def calculate_effective_sampling_ratio(profile: dict) -> tuple[float, float]:
    """
    Calculate the effective sampling ratio between rarest and most common words.

    Returns:
        (min_ratio, max_ratio): Minimum and maximum sampling weight ratios
    """
    if profile.get("strategy") != "inverse_sqrt_freq":
        return (1.0, 1.0)  # Can't calculate for other strategies

    freq_power = profile.get("freq_power", 0.5)
    rare_boost = profile.get("rare_word_boost", 1.0)
    max_factor = profile.get("max_weight_factor", 10.0)

    # Weight for word with frequency 1
    rare_weight = (1 ** -freq_power) * rare_boost

    # Weight for word with frequency 50000
    common_weight = 50000 ** -freq_power

    # Apply max factor capping
    if max_factor > 0:
        rare_weight = min(rare_weight, max_factor)
        common_weight = max(common_weight, 1.0 / max_factor)

    ratio = rare_weight / common_weight
    return (1.0 / max_factor, min(ratio, max_factor))


def recommend_profile_sequence():
    """
    Recommend a training sequence that gradually introduces rare words
    without breaking common word performance.
    """

    sequence = [
        ("production_safe", 5, "Establish baseline without disruption"),
        ("vocabulary_coverage", 3, "Ensure all words seen at least once"),
        ("mild_rare_boost", 5, "Gentle improvement for rare words"),
        ("length_balanced", 3, "Improve longer word handling"),
        ("stratified_balanced", 5, "Ensure balanced coverage"),
        ("production_safe", 2, "Stabilize for deployment"),
    ]

    print("RECOMMENDED TRAINING SEQUENCE")
    print("=" * 60)
    print("\nThis sequence gradually improves rare word handling without")
    print("causing the model to hallucinate uncommon words inappropriately.\n")

    total_epochs = 0
    for profile, epochs, reason in sequence:
        total_epochs += epochs
        config = BALANCED_PROFILES[profile]
        min_ratio, max_ratio = calculate_effective_sampling_ratio(config)

        print(f"Stage: {profile} ({epochs} epochs)")
        print(f"  Reason: {reason}")
        print(f"  Sampling ratio (rare:common): {max_ratio:.1f}x")
        print(f"  Description: {config['description']}")
        print()

    print(f"Total epochs: {total_epochs}")
    print("\nThis approach ensures:")
    print("- No extreme oversampling that breaks common words")
    print("- Gradual improvement in rare word coverage")
    print("- Model remains stable and production-ready")
    print("- Easy to roll back if metrics degrade")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BALANCED SAMPLING PROFILES")
    print("=" * 60)

    for name, config in BALANCED_PROFILES.items():
        min_ratio, max_ratio = calculate_effective_sampling_ratio(config)
        print(f"\n{name}:")
        print(f"  Max sampling ratio: {max_ratio:.1f}x")
        print(f"  Description: {config['description']}")

    print("\n" + "=" * 60)
    recommend_profile_sequence()