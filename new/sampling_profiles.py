#!/usr/bin/env python3
"""
Sampling profiles for frequency-aware training.

These profiles address the massive frequency imbalance in natural language where
common words like "the", "and", "to" appear thousands of times more frequently
than other words, which would cause training to plateau without intervention.
"""

SAMPLING_PROFILES = {
    # === FREQUENCY-BASED PROFILES ===

    "ultra_common_suppressed": {
        # Heavily suppress the most common words (top 100 by frequency)
        # to prevent them from dominating training
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.9,  # Strong inverse weighting
        "length_power": 0.0,  # Ignore word length
        "max_frequency": 100000,  # Focus on words that appear < 100k times
        "rare_frequency_threshold": 10000,
        "rare_word_boost": 0.5,  # Actually suppress "rare" (which are still common)
        "max_weight_factor": 5.0,
        "min_word_length": 1,
        "description": "Suppress ultra-common words (the, and, to, etc.)"
    },

    "common_balanced": {
        # Balance common words (rank 100-1000) with moderate suppression
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.6,
        "length_power": 0.2,
        "min_frequency": 1000,  # Skip ultra-rare
        "max_frequency": 50000,  # Skip ultra-common
        "rare_frequency_threshold": 5000,
        "rare_word_boost": 1.5,
        "max_weight_factor": 8.0,
        "min_word_length": 2,
        "description": "Balance common words (rank 100-1000)"
    },

    "medium_frequency": {
        # Focus on medium frequency words (rank 1000-10000)
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.3,
        "min_frequency": 100,
        "max_frequency": 5000,
        "rare_frequency_threshold": 500,
        "rare_word_boost": 2.0,
        "max_weight_factor": 10.0,
        "min_word_length": 3,
        "description": "Focus on medium frequency words"
    },

    "rare_focused": {
        # Heavily boost rare words (rank 10000+)
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.7,
        "length_power": 0.5,
        "max_frequency": 1000,  # Only words appearing < 1000 times
        "rare_frequency_threshold": 100,
        "rare_word_boost": 4.0,
        "max_weight_factor": 15.0,
        "min_word_length": 2,
        "description": "Boost rare words heavily"
    },

    "ultra_rare_boost": {
        # Extreme boost for the rarest words
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.9,
        "length_power": 0.7,
        "max_frequency": 100,  # Only words appearing < 100 times
        "rare_frequency_threshold": 50,
        "rare_word_boost": 8.0,
        "max_weight_factor": 20.0,
        "min_word_length": 3,
        "description": "Extreme boost for ultra-rare words"
    },

    # === LENGTH-BASED PROFILES ===

    "short_words": {
        # Focus on 2-4 character words
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.4,
        "length_power": -0.5,  # Negative to boost short words
        "rare_frequency_threshold": 1000,
        "rare_word_boost": 2.0,
        "max_weight_factor": 10.0,
        "min_word_length": 2,
        "max_word_length": 4,
        "description": "Focus on short words (2-4 chars)"
    },

    "medium_words": {
        # Focus on 5-7 character words
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.0,
        "rare_frequency_threshold": 500,
        "rare_word_boost": 2.5,
        "max_weight_factor": 10.0,
        "min_word_length": 5,
        "max_word_length": 7,
        "description": "Focus on medium words (5-7 chars)"
    },

    "long_words": {
        # Focus on 8+ character words
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.6,
        "length_power": 0.8,
        "rare_frequency_threshold": 200,
        "rare_word_boost": 3.0,
        "max_weight_factor": 12.0,
        "min_word_length": 8,
        "description": "Focus on long words (8+ chars)"
    },

    # === BALANCED PROFILES ===

    "uniform": {
        # No weighting - uniform sampling
        "strategy": "none",
        "description": "Uniform sampling (no weighting)"
    },

    "sqrt_balanced": {
        # Moderate sqrt-based balancing
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.0,
        "rare_frequency_threshold": 0,  # No special rare word handling
        "rare_word_boost": 1.0,
        "max_weight_factor": 10.0,
        "min_word_length": 1,
        "description": "Square root frequency balancing"
    },

    "production_balanced": {
        # Well-tested production settings
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.55,
        "length_power": 0.3,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 2.5,
        "max_weight_factor": 12.0,
        "min_word_length": 2,
        "description": "Production-tested balanced settings"
    },

    # === CURRICULUM STAGES ===

    "curriculum_stage1": {
        # Stage 1: Master common patterns first
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.3,  # Mild suppression of ultra-common
        "length_power": 0.0,
        "min_frequency": 1000,  # Skip rare words initially
        "rare_frequency_threshold": 10000,
        "rare_word_boost": 1.2,
        "max_weight_factor": 5.0,
        "min_word_length": 2,
        "max_word_length": 6,
        "description": "Stage 1: Common patterns (words length 2-6, freq > 1000)"
    },

    "curriculum_stage2": {
        # Stage 2: Expand to medium frequency
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.2,
        "min_frequency": 100,
        "max_frequency": 10000,
        "rare_frequency_threshold": 500,
        "rare_word_boost": 2.0,
        "max_weight_factor": 10.0,
        "min_word_length": 3,
        "description": "Stage 2: Medium frequency expansion"
    },

    "curriculum_stage3": {
        # Stage 3: Include rare words
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.7,
        "length_power": 0.4,
        "max_frequency": 1000,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 3.5,
        "max_weight_factor": 15.0,
        "min_word_length": 2,
        "description": "Stage 3: Include rare words"
    },

    "curriculum_stage4": {
        # Stage 4: Final balanced training
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.6,
        "length_power": 0.3,
        "rare_frequency_threshold": 200,
        "rare_word_boost": 2.5,
        "max_weight_factor": 12.0,
        "min_word_length": 1,
        "description": "Stage 4: Final balanced refinement"
    },

    # === VALIDATION PROFILES ===

    "validation_balanced": {
        # Balanced validation sampling
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.5,
        "length_power": 0.0,
        "rare_frequency_threshold": 0,
        "rare_word_boost": 1.0,
        "max_weight_factor": 5.0,
        "min_word_length": 1,
        "description": "Balanced validation sampling"
    },

    "validation_challenging": {
        # Focus validation on challenging cases
        "strategy": "inverse_sqrt_freq",
        "freq_power": 0.7,
        "length_power": 0.5,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 3.0,
        "max_weight_factor": 10.0,
        "min_word_length": 3,
        "description": "Focus on challenging validation cases"
    }
}

# === ALIASES FOR COMPREHENSIVE RUNNER ===
# Map runner profile names to existing profiles so orchestration scripts work out of the box.
ALIASES = {
    "short_common": {
        **SAMPLING_PROFILES["short_words"],
        "min_frequency": 1000,
        "description": "Alias: short words with common-bias",
    },
    "medium_balanced": {
        **SAMPLING_PROFILES["medium_words"],
        "description": "Alias: medium words (balanced)",
    },
    "base_random": {
        **SAMPLING_PROFILES["uniform"],
        "description": "Alias: uniform sampling",
    },
    "rare_words": {
        **SAMPLING_PROFILES["rare_focused"],
        "description": "Alias: rare-focused",
    },
    "very_rare": {
        **SAMPLING_PROFILES["ultra_rare_boost"],
        "description": "Alias: ultra-rare boost",
    },
    "high_confusion": {
        **SAMPLING_PROFILES["production_balanced"],
        "description": "Alias: high confusion ≈ production balanced",
    },
    "production_current": {
        **SAMPLING_PROFILES["production_balanced"],
        "description": "Alias: production current",
    },
    "validation_current": {
        **SAMPLING_PROFILES["validation_balanced"],
        "description": "Alias: validation current",
    },
}

SAMPLING_PROFILES.update(ALIASES)


def get_profile(name: str) -> dict:
    """Get a sampling profile by name."""
    if name not in SAMPLING_PROFILES:
        available = ", ".join(sorted(SAMPLING_PROFILES.keys()))
        raise ValueError(
            f"Unknown profile '{name}'. Available profiles: {available}"
        )
    return SAMPLING_PROFILES[name].copy()


def list_profiles() -> list:
    """List all available profile names."""
    return sorted(SAMPLING_PROFILES.keys())


def get_profile_description(name: str) -> str:
    """Get description for a profile."""
    profile = get_profile(name)
    return profile.get("description", "No description available")


if __name__ == "__main__":
    print("Available sampling profiles:")
    print("=" * 60)
    for name in list_profiles():
        desc = get_profile_description(name)
        print(f"  {name:25s} - {desc}")
