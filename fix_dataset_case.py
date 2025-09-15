#!/usr/bin/env python3
"""
Fix dataset by converting all words to lowercase to match vocabulary.
This eliminates <unk> tokens caused by uppercase letters.
"""

import json
import os

def fix_dataset(input_path, output_path):
    """Convert all words in dataset to lowercase."""

    fixed_count = 0
    total_count = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            original_word = data['word']
            data['word'] = original_word.lower()

            if original_word != data['word']:
                fixed_count += 1

            total_count += 1
            fout.write(json.dumps(data) + '\n')

    print(f"Processed {total_count} entries")
    print(f"Fixed {fixed_count} entries with uppercase ({100*fixed_count/total_count:.1f}%)")

def main():
    # Fix training data
    print("Fixing training data...")
    fix_dataset('data/train_final_train.jsonl', 'data/train_final_train_fixed.jsonl')

    # Fix validation data
    print("\nFixing validation data...")
    fix_dataset('data/train_final_val.jsonl', 'data/train_final_val_fixed.jsonl')

    # Backup originals and replace with fixed versions
    print("\nBacking up original files and replacing with fixed versions...")
    os.rename('data/train_final_train.jsonl', 'data/train_final_train_original.jsonl')
    os.rename('data/train_final_val.jsonl', 'data/train_final_val_original.jsonl')
    os.rename('data/train_final_train_fixed.jsonl', 'data/train_final_train.jsonl')
    os.rename('data/train_final_val_fixed.jsonl', 'data/train_final_val.jsonl')

    print("\n✓ Dataset fixed! All words are now lowercase.")
    print("  Original files backed up with '_original' suffix")

if __name__ == '__main__':
    main()