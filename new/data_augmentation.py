#!/usr/bin/env python3
"""
Data Augmentation for Swipe Gesture Training

Implements various augmentation techniques to improve model robustness,
especially for rare words where we have limited training examples.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class SwipeAugmentation:
    """Augmentation techniques for swipe gesture data."""

    def __init__(
        self,
        noise_std: float = 0.02,
        time_warp_factor: float = 0.1,
        spatial_shift_range: float = 0.05,
        speed_variation: float = 0.2,
        enable_for_rare_only: bool = True,
        rare_threshold: int = 50,
    ):
        """
        Initialize augmentation parameters.

        Args:
            noise_std: Standard deviation for Gaussian noise
            time_warp_factor: Maximum time warping factor
            spatial_shift_range: Maximum spatial shift in normalized coords
            speed_variation: Speed variation factor (0.2 = ±20%)
            enable_for_rare_only: Only augment rare words
            rare_threshold: Words with freq <= this are considered rare
        """
        self.noise_std = noise_std
        self.time_warp_factor = time_warp_factor
        self.spatial_shift_range = spatial_shift_range
        self.speed_variation_factor = speed_variation
        self.enable_for_rare_only = enable_for_rare_only
        self.rare_threshold = rare_threshold

        # Word frequency cache (will be populated by training script)
        self.word_frequencies: Dict[str, int] = {}

    def set_word_frequencies(self, frequencies: Dict[str, int]):
        """Set word frequency dictionary for determining rare words."""
        self.word_frequencies = frequencies

    def should_augment(self, word: str) -> bool:
        """Determine if a word should be augmented."""
        if not self.enable_for_rare_only:
            return True

        if not self.word_frequencies:
            return True  # Augment if no frequency info

        freq = self.word_frequencies.get(word, 0)
        return freq <= self.rare_threshold

    def add_noise(self, points: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to spatial coordinates."""
        if len(points) == 0:
            return points

        noise = np.random.normal(0, self.noise_std, (len(points), 2))
        points_aug = points.copy()
        points_aug[:, :2] += noise  # Only add to x,y, not time

        # Clip to valid range [-1, 1]
        points_aug[:, :2] = np.clip(points_aug[:, :2], -1.0, 1.0)
        return points_aug

    def time_warp(self, points: np.ndarray) -> np.ndarray:
        """Apply random time warping to gesture timing."""
        if len(points) <= 2:
            return points

        points_aug = points.copy()

        # Generate random warp points
        n_warp = min(3, len(points) // 10)  # Max 3 warp points
        if n_warp < 1:
            return points

        warp_indices = sorted(random.sample(range(1, len(points) - 1), n_warp))

        # Apply time warping
        for idx in warp_indices:
            warp = 1.0 + random.uniform(-self.time_warp_factor, self.time_warp_factor)
            # Warp timing for this segment
            if idx > 0:
                segment_duration = points_aug[idx, 2] - points_aug[idx-1, 2]
                points_aug[idx, 2] = points_aug[idx-1, 2] + segment_duration * warp

        # Normalize times to maintain gesture duration
        if points_aug[-1, 2] > 0:
            time_scale = points[-1, 2] / points_aug[-1, 2]
            points_aug[:, 2] *= time_scale

        return points_aug

    def spatial_shift(self, points: np.ndarray) -> np.ndarray:
        """Apply global spatial shift to entire gesture."""
        if len(points) == 0:
            return points

        shift_x = random.uniform(-self.spatial_shift_range, self.spatial_shift_range)
        shift_y = random.uniform(-self.spatial_shift_range, self.spatial_shift_range)

        points_aug = points.copy()
        points_aug[:, 0] += shift_x
        points_aug[:, 1] += shift_y

        # Clip to valid range
        points_aug[:, :2] = np.clip(points_aug[:, :2], -1.0, 1.0)
        return points_aug

    def speed_variation(self, points: np.ndarray) -> np.ndarray:
        """Vary the overall speed of the gesture."""
        if len(points) <= 1:
            return points

        speed_factor = 1.0 + random.uniform(-self.speed_variation_factor, self.speed_variation_factor)

        points_aug = points.copy()
        points_aug[:, 2] *= speed_factor

        return points_aug

    def elastic_deformation(self, points: np.ndarray, alpha: float = 0.1, sigma: float = 0.05) -> np.ndarray:
        """Apply elastic deformation to swipe path."""
        if len(points) <= 3:
            return points

        points_aug = points.copy()

        # Generate smooth random displacement field
        n_control = min(5, len(points) // 5)
        if n_control < 2:
            return points

        control_indices = np.linspace(0, len(points) - 1, n_control).astype(int)

        # Random displacements at control points
        dx = np.random.randn(n_control) * sigma
        dy = np.random.randn(n_control) * sigma

        # Interpolate displacements for all points
        all_dx = np.interp(range(len(points)), control_indices, dx) * alpha
        all_dy = np.interp(range(len(points)), control_indices, dy) * alpha

        points_aug[:, 0] += all_dx
        points_aug[:, 1] += all_dy

        # Clip to valid range
        points_aug[:, :2] = np.clip(points_aug[:, :2], -1.0, 1.0)
        return points_aug

    def augment(self, sample: Dict, augmentation_prob: float = 0.5) -> Dict:
        """
        Apply random augmentations to a training sample.

        Args:
            sample: Training sample with 'word' and 'points'
            augmentation_prob: Probability of applying augmentation

        Returns:
            Augmented sample
        """
        word = sample['word']

        # Check if we should augment this word
        if not self.should_augment(word):
            return sample

        if random.random() > augmentation_prob:
            return sample

        # Convert points to numpy array
        points = np.array([[p['x'], p['y'], p['t']] for p in sample['points']])

        # Apply random subset of augmentations
        augmentations = [
            (self.add_noise, 0.7),
            (self.time_warp, 0.3),
            (self.spatial_shift, 0.5),
            (self.speed_variation, 0.4),
            (lambda p: self.elastic_deformation(p), 0.3),
        ]

        # Apply augmentations
        for aug_func, prob in augmentations:
            if random.random() < prob:
                points = aug_func(points)

        # Convert back to sample format
        augmented_sample = sample.copy()
        augmented_sample['points'] = [
            {'x': float(p[0]), 'y': float(p[1]), 't': int(p[2])}
            for p in points
        ]

        # Mark as augmented
        augmented_sample['augmented'] = True

        return augmented_sample


class MixupAugmentation:
    """
    Mixup augmentation for swipe gestures.
    Blends two gestures at the feature level.
    """

    def __init__(self, alpha: float = 0.2, enable_for_rare_only: bool = True):
        """
        Initialize mixup parameters.

        Args:
            alpha: Beta distribution parameter for mixing ratio
            enable_for_rare_only: Only apply to rare words
        """
        self.alpha = alpha
        self.enable_for_rare_only = enable_for_rare_only
        self.augmenter = None  # Will share SwipeAugmentation instance

    def set_augmenter(self, augmenter: SwipeAugmentation):
        """Set reference to SwipeAugmentation for rare word checking."""
        self.augmenter = augmenter

    def mixup_features(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
        lambda_val: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to feature tensors.

        Args:
            features1: First batch of features [B, T, F]
            features2: Second batch of features [B, T, F]
            labels1: First batch of labels
            labels2: Second batch of labels
            lambda_val: Mixing ratio (if None, sample from Beta)

        Returns:
            Mixed features, mixed labels (one-hot), mixing ratio
        """
        if lambda_val is None:
            lambda_val = np.random.beta(self.alpha, self.alpha)

        # Mix features
        mixed_features = lambda_val * features1 + (1 - lambda_val) * features2

        # For labels, we'll return both with the lambda for loss calculation
        return mixed_features, labels1, labels2, lambda_val


class CutMixAugmentation:
    """
    CutMix augmentation adapted for sequential swipe data.
    Replaces a segment of one gesture with a segment from another.
    """

    def __init__(self, alpha: float = 1.0, enable_for_rare_only: bool = True):
        """
        Initialize cutmix parameters.

        Args:
            alpha: Parameter for segment size distribution
            enable_for_rare_only: Only apply to rare words
        """
        self.alpha = alpha
        self.enable_for_rare_only = enable_for_rare_only
        self.augmenter = None

    def set_augmenter(self, augmenter: SwipeAugmentation):
        """Set reference to SwipeAugmentation for rare word checking."""
        self.augmenter = augmenter

    def cutmix_features(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to sequential features.

        Args:
            features1: First batch of features [B, T, F]
            features2: Second batch of features [B, T, F]
            labels1: First batch of labels
            labels2: Second batch of labels

        Returns:
            Mixed features, labels for loss calculation, mixing ratio
        """
        batch_size, seq_len, feat_dim = features1.shape

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Calculate cut size
        cut_len = int(seq_len * (1 - lam))

        if cut_len == 0:
            return features1, labels1, labels2, 1.0

        # Random start position
        cut_start = random.randint(0, seq_len - cut_len)
        cut_end = cut_start + cut_len

        # Apply CutMix
        mixed_features = features1.clone()
        mixed_features[:, cut_start:cut_end, :] = features2[:, cut_start:cut_end, :]

        # Calculate actual mixing ratio based on cut size
        actual_lam = 1 - (cut_len / seq_len)

        return mixed_features, labels1, labels2, actual_lam


def create_augmentation_pipeline(
    config: Optional[Dict] = None,
    word_frequencies: Optional[Dict[str, int]] = None
) -> SwipeAugmentation:
    """
    Create augmentation pipeline with configuration.

    Args:
        config: Augmentation configuration dictionary
        word_frequencies: Word frequency dictionary

    Returns:
        Configured SwipeAugmentation instance
    """
    if config is None:
        config = {}

    augmenter = SwipeAugmentation(
        noise_std=config.get('noise_std', 0.02),
        time_warp_factor=config.get('time_warp_factor', 0.1),
        spatial_shift_range=config.get('spatial_shift_range', 0.05),
        speed_variation=config.get('speed_variation', 0.2),
        enable_for_rare_only=config.get('enable_for_rare_only', True),
        rare_threshold=config.get('rare_threshold', 50),
    )

    if word_frequencies:
        augmenter.set_word_frequencies(word_frequencies)

    return augmenter