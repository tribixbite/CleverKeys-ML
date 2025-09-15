#!/usr/bin/env python3
"""Deep dataset analysis to identify data quality issues causing NaN losses."""

import json
import numpy as np
import math
from collections import defaultdict, Counter
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file and return list of data samples."""
    samples = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                sample['line_num'] = line_num
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    return samples

def analyze_coordinates(samples):
    """Analyze coordinate values for anomalies."""
    all_x = []
    all_y = []
    all_t = []
    invalid_coords = []
    out_of_bounds = []
    
    for sample in samples:
        points = sample.get('points', [])
        for i, point in enumerate(points):
            x, y, t = point.get('x', 0), point.get('y', 0), point.get('t', 0)
            
            # Check for invalid values
            if math.isnan(x) or math.isnan(y) or math.isnan(t):
                invalid_coords.append((sample['line_num'], sample.get('word', ''), i, 'NaN'))
            elif math.isinf(x) or math.isinf(y) or math.isinf(t):
                invalid_coords.append((sample['line_num'], sample.get('word', ''), i, 'Inf'))
            
            # Check bounds (assuming normalized 0-1 for keyboard)
            if x < -0.1 or x > 1.1 or y < -0.1 or y > 1.1:
                out_of_bounds.append((sample['line_num'], sample.get('word', ''), i, f'x={x:.3f}, y={y:.3f}'))
            
            all_x.append(x)
            all_y.append(y)
            all_t.append(t)
    
    return {
        'x_stats': {
            'min': np.min(all_x) if all_x else None,
            'max': np.max(all_x) if all_x else None,
            'mean': np.mean(all_x) if all_x else None,
            'std': np.std(all_x) if all_x else None,
            'nan_count': sum(1 for x in all_x if math.isnan(x)),
            'inf_count': sum(1 for x in all_x if math.isinf(x))
        },
        'y_stats': {
            'min': np.min(all_y) if all_y else None,
            'max': np.max(all_y) if all_y else None,
            'mean': np.mean(all_y) if all_y else None,
            'std': np.std(all_y) if all_y else None,
            'nan_count': sum(1 for y in all_y if math.isnan(y)),
            'inf_count': sum(1 for y in all_y if math.isinf(y))
        },
        't_stats': {
            'min': np.min(all_t) if all_t else None,
            'max': np.max(all_t) if all_t else None,
            'mean': np.mean(all_t) if all_t else None,
            'std': np.std(all_t) if all_t else None,
            'nan_count': sum(1 for t in all_t if math.isnan(t)),
            'inf_count': sum(1 for t in all_t if math.isinf(t))
        },
        'invalid_coords': invalid_coords,
        'out_of_bounds': out_of_bounds
    }

def analyze_sequences(samples):
    """Analyze sequence lengths and patterns."""
    seq_lengths = []
    word_lengths = []
    length_ratios = []
    empty_sequences = []
    very_short = []
    very_long = []
    
    for sample in samples:
        points = sample.get('points', [])
        word = sample.get('word', '')
        
        seq_len = len(points)
        word_len = len(word)
        
        seq_lengths.append(seq_len)
        word_lengths.append(word_len)
        
        if seq_len == 0:
            empty_sequences.append((sample['line_num'], word))
        elif seq_len < 5:
            very_short.append((sample['line_num'], word, seq_len))
        elif seq_len > 500:
            very_long.append((sample['line_num'], word, seq_len))
        
        if word_len > 0:
            ratio = seq_len / word_len
            length_ratios.append(ratio)
    
    return {
        'seq_length_stats': {
            'min': min(seq_lengths) if seq_lengths else None,
            'max': max(seq_lengths) if seq_lengths else None,
            'mean': np.mean(seq_lengths) if seq_lengths else None,
            'median': np.median(seq_lengths) if seq_lengths else None,
            'std': np.std(seq_lengths) if seq_lengths else None,
            'percentiles': np.percentile(seq_lengths, [25, 50, 75, 90, 95, 99]) if seq_lengths else None
        },
        'word_length_stats': {
            'min': min(word_lengths) if word_lengths else None,
            'max': max(word_lengths) if word_lengths else None,
            'mean': np.mean(word_lengths) if word_lengths else None,
            'median': np.median(word_lengths) if word_lengths else None
        },
        'ratio_stats': {
            'min': min(length_ratios) if length_ratios else None,
            'max': max(length_ratios) if length_ratios else None,
            'mean': np.mean(length_ratios) if length_ratios else None,
            'median': np.median(length_ratios) if length_ratios else None
        },
        'empty_sequences': empty_sequences,
        'very_short': very_short,
        'very_long': very_long
    }

def analyze_velocity_acceleration(samples):
    """Analyze velocity and acceleration patterns that might cause NaN in training."""
    velocity_issues = []
    acceleration_issues = []
    time_issues = []
    
    for sample in samples:
        points = sample.get('points', [])
        word = sample.get('word', '')
        
        if len(points) < 2:
            continue
        
        # Calculate velocities and accelerations
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            
            dt = curr.get('t', 0) - prev.get('t', 0)
            dx = curr.get('x', 0) - prev.get('x', 0)
            dy = curr.get('y', 0) - prev.get('y', 0)
            
            # Check for time issues
            if dt <= 0:
                time_issues.append((sample['line_num'], word, i, f'dt={dt}'))
                continue
            
            # Calculate velocity
            vx = dx / dt if dt > 0 else 0
            vy = dy / dt if dt > 0 else 0
            velocity = math.sqrt(vx**2 + vy**2)
            
            # Check for extreme velocities
            if velocity > 100:  # Unreasonably high velocity
                velocity_issues.append((sample['line_num'], word, i, f'v={velocity:.2f}'))
            
            # Calculate acceleration if we have enough points
            if i >= 2:
                prev2 = points[i-2]
                dt2 = prev.get('t', 0) - prev2.get('t', 0)
                if dt2 > 0:
                    dx2 = prev.get('x', 0) - prev2.get('x', 0)
                    dy2 = prev.get('y', 0) - prev2.get('y', 0)
                    vx2 = dx2 / dt2
                    vy2 = dy2 / dt2
                    
                    ax = (vx - vx2) / dt if dt > 0 else 0
                    ay = (vy - vy2) / dt if dt > 0 else 0
                    acceleration = math.sqrt(ax**2 + ay**2)
                    
                    if acceleration > 1000:  # Unreasonably high acceleration
                        acceleration_issues.append((sample['line_num'], word, i, f'a={acceleration:.2f}'))
    
    return {
        'velocity_issues': velocity_issues,
        'acceleration_issues': acceleration_issues,
        'time_issues': time_issues
    }

def analyze_duplicates_and_patterns(samples):
    """Analyze for duplicate entries and suspicious patterns."""
    word_counts = Counter()
    duplicate_gestures = defaultdict(list)
    gesture_hashes = {}
    
    for sample in samples:
        word = sample.get('word', '')
        word_counts[word] += 1
        
        # Create a hash of the gesture for duplicate detection
        points = sample.get('points', [])
        if points:
            # Create a simplified hash of the gesture
            gesture_str = json.dumps([(round(p.get('x', 0), 3), round(p.get('y', 0), 3)) for p in points[:10]])
            if gesture_str in gesture_hashes:
                duplicate_gestures[gesture_str].append((sample['line_num'], word))
            else:
                gesture_hashes[gesture_str] = (sample['line_num'], word)
    
    # Find exact duplicates
    exact_duplicates = {k: v for k, v in duplicate_gestures.items() if len(v) > 0}
    
    return {
        'top_words': word_counts.most_common(20),
        'unique_words': len(word_counts),
        'total_samples': sum(word_counts.values()),
        'exact_duplicate_gestures': len(exact_duplicates),
        'duplicate_examples': list(exact_duplicates.values())[:10] if exact_duplicates else []
    }

def analyze_data_distribution(samples):
    """Analyze overall data distribution and balance."""
    char_distribution = Counter()
    first_char_distribution = Counter()
    last_char_distribution = Counter()
    
    for sample in samples:
        word = sample.get('word', '')
        for char in word:
            char_distribution[char] += 1
        if word:
            first_char_distribution[word[0]] += 1
            last_char_distribution[word[-1]] += 1
    
    return {
        'char_distribution': dict(char_distribution.most_common()),
        'first_char_distribution': dict(first_char_distribution.most_common(10)),
        'last_char_distribution': dict(last_char_distribution.most_common(10))
    }

def check_specific_problematic_samples(samples, n=10):
    """Identify specific samples that might cause issues."""
    problematic = []
    
    for sample in samples[:n]:  # Check first n samples
        issues = []
        points = sample.get('points', [])
        word = sample.get('word', '')
        
        # Check various issues
        if len(points) == 0:
            issues.append('empty_points')
        elif len(points) < 5:
            issues.append(f'very_short ({len(points)} points)')
        
        if not word:
            issues.append('empty_word')
        
        # Check for time monotonicity
        if len(points) > 1:
            times = [p.get('t', 0) for p in points]
            if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
                issues.append('non_monotonic_time')
        
        # Check for static gestures (no movement)
        if len(points) > 1:
            x_vals = [p.get('x', 0) for p in points]
            y_vals = [p.get('y', 0) for p in points]
            if np.std(x_vals) < 0.01 and np.std(y_vals) < 0.01:
                issues.append('static_gesture')
        
        if issues:
            problematic.append({
                'line': sample['line_num'],
                'word': word,
                'seq_len': len(points),
                'issues': issues
            })
    
    return problematic

def main():
    # Paths to data files
    train_path = Path('data/train_final_train.jsonl')
    val_path = Path('data/train_final_val.jsonl')
    
    print("=" * 80)
    print("DEEP DATASET ANALYSIS FOR NaN LOSS DEBUGGING")
    print("=" * 80)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)
    print(f"✓ Loaded {len(train_samples)} training samples")
    print(f"✓ Loaded {len(val_samples)} validation samples")
    
    # Analyze both datasets
    for dataset_name, samples in [('TRAINING', train_samples), ('VALIDATION', val_samples)]:
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset_name} DATA")
        print(f"{'='*80}")
        
        # 1. Coordinate Analysis
        print("\n📊 Coordinate Analysis:")
        coord_analysis = analyze_coordinates(samples)
        print(f"  X coordinates: min={coord_analysis['x_stats']['min']:.3f}, max={coord_analysis['x_stats']['max']:.3f}, "
              f"mean={coord_analysis['x_stats']['mean']:.3f}, std={coord_analysis['x_stats']['std']:.3f}")
        print(f"  Y coordinates: min={coord_analysis['y_stats']['min']:.3f}, max={coord_analysis['y_stats']['max']:.3f}, "
              f"mean={coord_analysis['y_stats']['mean']:.3f}, std={coord_analysis['y_stats']['std']:.3f}")
        print(f"  T values: min={coord_analysis['t_stats']['min']:.3f}, max={coord_analysis['t_stats']['max']:.3f}, "
              f"mean={coord_analysis['t_stats']['mean']:.3f}, std={coord_analysis['t_stats']['std']:.3f}")
        
        if coord_analysis['invalid_coords']:
            print(f"\n  ⚠️  Found {len(coord_analysis['invalid_coords'])} invalid coordinates!")
            for item in coord_analysis['invalid_coords'][:5]:
                print(f"    Line {item[0]}, word '{item[1]}', point {item[2]}: {item[3]}")
        
        if coord_analysis['out_of_bounds']:
            print(f"\n  ⚠️  Found {len(coord_analysis['out_of_bounds'])} out-of-bounds coordinates!")
            for item in coord_analysis['out_of_bounds'][:5]:
                print(f"    Line {item[0]}, word '{item[1]}', point {item[2]}: {item[3]}")
        
        # 2. Sequence Analysis
        print("\n📏 Sequence Length Analysis:")
        seq_analysis = analyze_sequences(samples)
        print(f"  Sequence lengths: min={seq_analysis['seq_length_stats']['min']}, "
              f"max={seq_analysis['seq_length_stats']['max']}, "
              f"mean={seq_analysis['seq_length_stats']['mean']:.1f}, "
              f"median={seq_analysis['seq_length_stats']['median']:.1f}")
        print(f"  Percentiles (25/50/75/90/95/99): {seq_analysis['seq_length_stats']['percentiles']}")
        print(f"  Word lengths: min={seq_analysis['word_length_stats']['min']}, "
              f"max={seq_analysis['word_length_stats']['max']}, "
              f"mean={seq_analysis['word_length_stats']['mean']:.1f}")
        print(f"  Points/char ratio: mean={seq_analysis['ratio_stats']['mean']:.1f}, "
              f"median={seq_analysis['ratio_stats']['median']:.1f}")
        
        if seq_analysis['empty_sequences']:
            print(f"\n  ⚠️  Found {len(seq_analysis['empty_sequences'])} empty sequences!")
            for item in seq_analysis['empty_sequences'][:5]:
                print(f"    Line {item[0]}: word '{item[1]}'")
        
        if seq_analysis['very_short']:
            print(f"\n  ⚠️  Found {len(seq_analysis['very_short'])} very short sequences (<5 points)!")
            for item in seq_analysis['very_short'][:5]:
                print(f"    Line {item[0]}: word '{item[1]}' has only {item[2]} points")
        
        if seq_analysis['very_long']:
            print(f"\n  ⚠️  Found {len(seq_analysis['very_long'])} very long sequences (>500 points)!")
            for item in seq_analysis['very_long'][:5]:
                print(f"    Line {item[0]}: word '{item[1]}' has {item[2]} points")
        
        # 3. Velocity/Acceleration Analysis
        print("\n⚡ Velocity/Acceleration Analysis:")
        motion_analysis = analyze_velocity_acceleration(samples)
        
        if motion_analysis['time_issues']:
            print(f"  ⚠️  Found {len(motion_analysis['time_issues'])} time issues (dt <= 0)!")
            for item in motion_analysis['time_issues'][:5]:
                print(f"    Line {item[0]}, word '{item[1]}', point {item[2]}: {item[3]}")
        
        if motion_analysis['velocity_issues']:
            print(f"  ⚠️  Found {len(motion_analysis['velocity_issues'])} extreme velocity issues!")
            for item in motion_analysis['velocity_issues'][:5]:
                print(f"    Line {item[0]}, word '{item[1]}', point {item[2]}: {item[3]}")
        
        if motion_analysis['acceleration_issues']:
            print(f"  ⚠️  Found {len(motion_analysis['acceleration_issues'])} extreme acceleration issues!")
            for item in motion_analysis['acceleration_issues'][:5]:
                print(f"    Line {item[0]}, word '{item[1]}', point {item[2]}: {item[3]}")
        
        # 4. Duplicates and Patterns
        print("\n🔍 Duplicate and Pattern Analysis:")
        pattern_analysis = analyze_duplicates_and_patterns(samples)
        print(f"  Unique words: {pattern_analysis['unique_words']}")
        print(f"  Total samples: {pattern_analysis['total_samples']}")
        print(f"  Exact duplicate gestures: {pattern_analysis['exact_duplicate_gestures']}")
        print(f"\n  Top 10 most frequent words:")
        for word, count in pattern_analysis['top_words'][:10]:
            print(f"    '{word}': {count} occurrences")
        
        # 5. Character Distribution
        print("\n🔤 Character Distribution:")
        dist_analysis = analyze_data_distribution(samples)
        print(f"  Unique characters: {len(dist_analysis['char_distribution'])}")
        print(f"  Character set: {sorted(dist_analysis['char_distribution'].keys())}")
        
        # 6. Check First Few Samples (these would be in first batch)
        print("\n🎯 First 10 Samples Analysis (First Training Batch):")
        problematic = check_specific_problematic_samples(samples, n=10)
        if problematic:
            print("  ⚠️  Problematic samples in first batch:")
            for p in problematic:
                print(f"    Line {p['line']}, word '{p['word']}', {p['seq_len']} points: {', '.join(p['issues'])}")
        else:
            print("  ✓ No obvious issues in first 10 samples")
    
    print("\n" + "="*80)
    print("CRITICAL FINDINGS FOR NaN LOSSES:")
    print("="*80)
    
    # Summarize critical issues
    print("\n🚨 Most likely causes of NaN losses:")
    print("1. Check for empty sequences or very short sequences in early batches")
    print("2. Check for time monotonicity issues (dt <= 0 causes division by zero)")
    print("3. Check for extreme velocities/accelerations from noisy data")
    print("4. Check for coordinate values outside expected bounds")
    print("5. Check for sequences longer than model's max length (200)")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()