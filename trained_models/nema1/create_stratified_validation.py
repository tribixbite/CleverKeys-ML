#!/usr/bin/env python3
"""
Create Stratified Validation Sets for CleverKeys Training

Creates multiple validation subsets to track performance on different word categories:
- Common words (high frequency)
- Rare words (low frequency)
- Long words (8+ characters)
- Overall balanced set
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StratifiedValidationCreator:
    """Creates stratified validation sets from training data."""

    def __init__(
        self,
        train_file: str = "../../data/train_final_train.jsonl",
        val_file: str = "../../data/train_final_val.jsonl",
        output_dir: str = "../../data/stratified_val/",
    ):
        """
        Initialize validation creator.

        Args:
            train_file: Path to training data (for frequency calculation)
            val_file: Path to validation data
            output_dir: Directory to save stratified validation sets
        """
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.word_frequencies = self._calculate_word_frequencies()
        self.val_samples = self._load_validation_data()

    def _calculate_word_frequencies(self) -> Dict[str, int]:
        """Calculate word frequencies from training data."""
        logger.info(f"Calculating word frequencies from {self.train_file}")

        word_counts = Counter()
        with open(self.train_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                word = sample['word']
                word_counts[word] += 1

        logger.info(f"Found {len(word_counts)} unique words in training data")
        return dict(word_counts)

    def _load_validation_data(self) -> List[Dict]:
        """Load validation data."""
        logger.info(f"Loading validation data from {self.val_file}")

        samples = []
        with open(self.val_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                # Add frequency information
                word = sample['word']
                sample['frequency'] = self.word_frequencies.get(word, 0)
                sample['word_length'] = len(word)
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} validation samples")
        return samples

    def create_common_words_subset(self, size: int = 2000, min_freq: int = 1000) -> List[Dict]:
        """
        Create validation subset of common words.

        Args:
            size: Number of samples to include
            min_freq: Minimum frequency to be considered common

        Returns:
            List of validation samples
        """
        common_samples = [s for s in self.val_samples if s['frequency'] >= min_freq]

        if len(common_samples) < size:
            logger.warning(f"Only {len(common_samples)} common word samples available")
            return common_samples

        return random.sample(common_samples, size)

    def create_rare_words_subset(self, size: int = 2000, max_freq: int = 50) -> List[Dict]:
        """
        Create validation subset of rare words.

        Args:
            size: Number of samples to include
            max_freq: Maximum frequency to be considered rare

        Returns:
            List of validation samples
        """
        rare_samples = [s for s in self.val_samples if s['frequency'] <= max_freq]

        if len(rare_samples) < size:
            logger.warning(f"Only {len(rare_samples)} rare word samples available")
            return rare_samples

        return random.sample(rare_samples, size)

    def create_long_words_subset(self, size: int = 2000, min_length: int = 8) -> List[Dict]:
        """
        Create validation subset of long words.

        Args:
            size: Number of samples to include
            min_length: Minimum word length

        Returns:
            List of validation samples
        """
        long_samples = [s for s in self.val_samples if s['word_length'] >= min_length]

        if len(long_samples) < size:
            logger.warning(f"Only {len(long_samples)} long word samples available")
            return long_samples

        return random.sample(long_samples, size)

    def create_balanced_subset(self, size: int = 3000) -> List[Dict]:
        """
        Create balanced validation subset with equal representation.

        Args:
            size: Total number of samples

        Returns:
            List of validation samples
        """
        # Divide into categories
        n_per_category = size // 4

        # Get samples from each category
        very_common = [s for s in self.val_samples if s['frequency'] > 5000]
        common = [s for s in self.val_samples if 1000 <= s['frequency'] <= 5000]
        uncommon = [s for s in self.val_samples if 100 <= s['frequency'] < 1000]
        rare = [s for s in self.val_samples if s['frequency'] < 100]

        balanced = []

        # Sample from each category
        for category, name in [(very_common, 'very_common'),
                               (common, 'common'),
                               (uncommon, 'uncommon'),
                               (rare, 'rare')]:
            if len(category) >= n_per_category:
                balanced.extend(random.sample(category, n_per_category))
            else:
                logger.warning(f"Category {name} has only {len(category)} samples")
                balanced.extend(category)

        # Fill remainder if needed
        remaining = size - len(balanced)
        if remaining > 0:
            unused = [s for s in self.val_samples if s not in balanced]
            if unused:
                balanced.extend(random.sample(unused, min(remaining, len(unused))))

        random.shuffle(balanced)
        return balanced

    def create_confusion_subset(self, size: int = 2000) -> List[Dict]:
        """
        Create validation subset of words that are commonly confused.
        (This is a placeholder - would need confusion matrix data)

        Args:
            size: Number of samples

        Returns:
            List of validation samples
        """
        # Common confusion patterns (would be better with actual data)
        confusion_patterns = [
            # Similar swipe patterns
            ('their', 'there'),
            ('your', 'you'),
            ('its', 'it'),
            ('too', 'to'),
            ('then', 'than'),
            ('affect', 'effect'),
            # Short words that are easily confused
            ('if', 'of'),
            ('in', 'on'),
            ('is', 'us'),
        ]

        confusion_words = set()
        for word1, word2 in confusion_patterns:
            confusion_words.add(word1)
            confusion_words.add(word2)

        confusion_samples = [s for s in self.val_samples
                            if s['word'] in confusion_words]

        # Add words with similar lengths and starting letters
        if len(confusion_samples) < size:
            # Group by first letter and length
            grouped = {}
            for s in self.val_samples:
                key = (s['word'][0], s['word_length'])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(s)

            # Add samples from groups with multiple words
            for key, group in grouped.items():
                if len(group) > 5:  # Groups likely to have confusion
                    confusion_samples.extend(random.sample(group, min(5, len(group))))

        if len(confusion_samples) > size:
            return random.sample(confusion_samples, size)
        return confusion_samples

    def save_subset(self, subset: List[Dict], name: str):
        """Save a validation subset to file."""
        output_file = self.output_dir / f"val_{name}.jsonl"

        # Remove frequency and word_length from saved data
        clean_subset = []
        for sample in subset:
            clean_sample = {
                'word': sample['word'],
                'points': sample['points']
            }
            clean_subset.append(clean_sample)

        with open(output_file, 'w') as f:
            for sample in clean_subset:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Saved {len(subset)} samples to {output_file}")

    def create_all_subsets(self):
        """Create all stratified validation subsets."""
        logger.info("Creating stratified validation subsets...")

        # Set random seed for reproducibility
        random.seed(42)

        # Create subsets
        subsets = {
            'common': self.create_common_words_subset(size=2000, min_freq=1000),
            'rare': self.create_rare_words_subset(size=2000, max_freq=50),
            'long': self.create_long_words_subset(size=2000, min_length=8),
            'balanced': self.create_balanced_subset(size=3000),
            'confusion': self.create_confusion_subset(size=1500),
        }

        # Save all subsets
        for name, subset in subsets.items():
            self.save_subset(subset, name)

        # Create summary statistics
        self._create_summary(subsets)

    def _create_summary(self, subsets: Dict[str, List[Dict]]):
        """Create summary statistics for all subsets."""
        summary = []
        summary.append("=" * 80)
        summary.append("STRATIFIED VALIDATION SETS SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        for name, subset in subsets.items():
            if not subset:
                continue

            # Calculate statistics
            frequencies = [s['frequency'] for s in subset]
            lengths = [s['word_length'] for s in subset]

            summary.append(f"## {name.upper()} Subset")
            summary.append(f"  - Samples: {len(subset)}")
            summary.append(f"  - Avg frequency: {sum(frequencies)/len(frequencies):.1f}")
            summary.append(f"  - Min frequency: {min(frequencies)}")
            summary.append(f"  - Max frequency: {max(frequencies)}")
            summary.append(f"  - Avg word length: {sum(lengths)/len(lengths):.1f}")
            summary.append(f"  - Min word length: {min(lengths)}")
            summary.append(f"  - Max word length: {max(lengths)}")

            # Sample words
            sample_words = random.sample([s['word'] for s in subset], min(10, len(subset)))
            summary.append(f"  - Sample words: {', '.join(sample_words)}")
            summary.append("")

        # Save summary
        summary_file = self.output_dir / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary))

        logger.info(f"Saved summary to {summary_file}")

        # Print summary
        print('\n'.join(summary))


def main():
    parser = argparse.ArgumentParser(description="Create stratified validation sets")
    parser.add_argument(
        '--train-file',
        default='../../data/train_final_train.jsonl',
        help='Path to training data'
    )
    parser.add_argument(
        '--val-file',
        default='../../data/train_final_val.jsonl',
        help='Path to validation data'
    )
    parser.add_argument(
        '--output-dir',
        default='../../data/stratified_val/',
        help='Directory to save stratified validation sets'
    )

    args = parser.parse_args()

    creator = StratifiedValidationCreator(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir
    )

    creator.create_all_subsets()


if __name__ == '__main__':
    main()