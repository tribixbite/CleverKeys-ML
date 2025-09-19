#!/usr/bin/env python3
"""
Train with specific sampling profile

Usage:
    python train_with_profile.py --profile rare_words
    python train_with_profile.py --profile base_random --compare-checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "nema1"))

from sampling_profiles import SAMPLING_PROFILES, get_profile
from omegaconf import DictConfig

def update_config_with_profile(config_dict: dict, profile_name: str) -> dict:
    """Update training config with sampling profile."""
    profile = get_profile(profile_name)

    # Remove description field
    profile = {k: v for k, v in profile.items() if k != "description"}

    # Update sampling config
    config_dict["sampling"] = profile

    # For validation profiles, also update validation config
    if "validation" in profile_name:
        val_profile = profile.copy()
        # Remove training-specific params from validation
        for key in ["strategy", "freq_power", "length_power", "min_word_length",
                    "max_word_length", "rare_frequency_threshold", "rare_word_boost",
                    "max_weight_factor", "max_frequency"]:
            val_profile.pop(key, None)
        config_dict["validation"].update(val_profile)

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train RNNT with specific sampling profile")
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(SAMPLING_PROFILES.keys()),
        default="production_current",
        help="Sampling profile to use"
    )
    parser.add_argument(
        "--compare-checkpoint",
        action="store_true",
        help="Use base_random to compare with old checkpoints"
    )
    parser.add_argument(
        "--export-config",
        type=str,
        help="Export config to JSON file instead of training"
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Run single batch for testing"
    )

    args = parser.parse_args()

    # Load base config
    from train_transducer_personalized import CONFIG

    # Apply profile
    config = CONFIG.copy()
    config = update_config_with_profile(config, args.profile)

    print(f"\n{'='*60}")
    print(f"Training with profile: {args.profile}")
    print(f"{'='*60}")

    profile_info = SAMPLING_PROFILES[args.profile]
    print(f"Description: {profile_info.get('description', 'No description')}")
    print(f"\nSampling parameters:")
    for key, value in config["sampling"].items():
        print(f"  {key}: {value}")

    if args.export_config:
        with open(args.export_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig exported to: {args.export_config}")
        return

    # Set environment for fast dev if requested
    if args.fast_dev:
        import os
        os.environ["FAST_DEV_RUN"] = "1"

    # Import and run training
    from train_transducer_personalized import main as train_main

    # Monkey-patch the config
    import train_transducer_personalized
    train_transducer_personalized.CONFIG = config

    # Run training
    train_main()


if __name__ == "__main__":
    main()