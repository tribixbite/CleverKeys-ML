import random
import sys
from tqdm import tqdm

def split_dataset(input_file: str, train_output: str, val_output: str, val_split_ratio: float = 0.05):
    """
    Shuffles and splits a .jsonl dataset into training and validation sets.

    Args:
        input_file: Path to the large source .jsonl file.
        train_output: Path to write the training split.
        val_output: Path to write the validation split.
        val_split_ratio: The fraction of data to use for validation (e.g., 0.05 for 5%).
    """
    print(f"Reading dataset from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        print("Please update the path to your 500k trace file.")
        sys.exit(1)

    print(f"Read {len(lines):,} lines. Shuffling data...")
    random.shuffle(lines)

    split_index = int(len(lines) * (1 - val_split_ratio))
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    print(f"Writing {len(train_lines):,} lines to {train_output}...")
    with open(train_output, 'w', encoding='utf-8') as f:
        for line in tqdm(train_lines, desc="Writing Train Split"):
            f.write(line)

    print(f"Writing {len(val_lines):,} lines to {val_output}...")
    with open(val_output, 'w', encoding='utf-8') as f:
        for line in tqdm(val_lines, desc="Writing Val Split"):
            f.write(line)

    print("\nDataset splitting complete!")
    print(f"Training set: {train_output}")
    print(f"Validation set: {val_output}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # IMPORTANT: Update this path to your actual 500k trace file!
    SOURCE_FILE = "../data/futo/train_final.jsonl" 
    
    TRAIN_DESTINATION = "../data/futo/train_final_train.jsonl"
    VAL_DESTINATION = "../data/futo/train_final_val.jsonl"
    # -------------------

    split_dataset(SOURCE_FILE, TRAIN_DESTINATION, VAL_DESTINATION)
