#!/usr/bin/env python3
"""Check what resample target is used for 91 points"""

def determine_resample_target(length: int) -> int:
    """From train_transducer_personalized.py"""
    resample_short_target = 56
    resample_long_target = 96
    resample_short_threshold = 48
    resample_long_threshold = 112

    if length < 20:
        return length

    if length <= resample_short_threshold:
        return resample_short_target
    elif length >= resample_long_threshold:
        return resample_long_target
    else:
        # Linear interpolation for smooth transition
        frac = (length - resample_short_threshold) / (resample_long_threshold - resample_short_threshold)
        target = resample_short_target + frac * (resample_long_target - resample_short_target)
        return int(target)

print(f"91 points -> {determine_resample_target(91)} frames")
print(f"48 points -> {determine_resample_target(48)} frames")
print(f"112 points -> {determine_resample_target(112)} frames")