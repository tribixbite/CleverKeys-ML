#!/usr/bin/env python3
"""
Compare keyboard layouts between training and web demo to find discrepancies
"""

def build_training_key_centers():
    """Exact key centers from training code"""
    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    centers = {}
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0  # Normalized to [0,1]
            y01 = (row_idx + 0.5) / 3.0   # Normalized to [0,1]
            # Training uses [-1,1] range
            x_train = x01 * 2.0 - 1.0
            y_train = y01 * 2.0 - 1.0
            centers[char] = (x_train, y_train, x01, y01)
    return centers


def build_web_key_positions():
    """Key positions used in web demo"""
    positions = {
        'q': (0.05, 0.25), 'w': (0.15, 0.25), 'e': (0.25, 0.25),
        'r': (0.35, 0.25), 't': (0.45, 0.25), 'y': (0.55, 0.25),
        'u': (0.65, 0.25), 'i': (0.75, 0.25), 'o': (0.85, 0.25),
        'p': (0.95, 0.25),
        'a': (0.10, 0.50), 's': (0.20, 0.50), 'd': (0.30, 0.50),
        'f': (0.40, 0.50), 'g': (0.50, 0.50), 'h': (0.60, 0.50),
        'j': (0.70, 0.50), 'k': (0.80, 0.50), 'l': (0.90, 0.50),
        'z': (0.20, 0.75), 'x': (0.30, 0.75), 'c': (0.40, 0.75),
        'v': (0.50, 0.75), 'b': (0.60, 0.75), 'n': (0.70, 0.75),
        'm': (0.80, 0.75)
    }
    return positions


def compare_layouts():
    """Compare the two layouts"""
    training = build_training_key_centers()
    web = build_web_key_positions()

    print("KEYBOARD LAYOUT COMPARISON")
    print("="*70)
    print(f"{'Char':<5} {'Training [-1,1]':<20} {'Training [0,1]':<20} {'Web [0,1]':<20} {'Match?':<10}")
    print("-"*70)

    for char in sorted(training.keys()):
        x_train, y_train, x01, y01 = training[char]
        if char in web:
            x_web, y_web = web[char]
            # Check if they match (with tolerance)
            match = abs(x01 - x_web) < 0.01 and abs(y01 - y_web) < 0.01
            match_str = "✓" if match else f"✗ Δx={x01-x_web:.3f}, Δy={y01-y_web:.3f}"

            print(f"{char:<5} ({x_train:>6.3f}, {y_train:>6.3f})  "
                  f"({x01:>6.3f}, {y01:>6.3f})  "
                  f"({x_web:>6.3f}, {y_web:>6.3f})  "
                  f"{match_str}")
        else:
            print(f"{char:<5} ({x_train:>6.3f}, {y_train:>6.3f})  "
                  f"({x01:>6.3f}, {y01:>6.3f})  "
                  f"{'NOT IN WEB':<20}")

    print("\nKEY INSIGHTS:")
    print("-"*70)

    # Check row-by-row
    rows = [
        ("qwertyuiop", "Top row"),
        ("asdfghjkl", "Middle row"),
        ("zxcvbnm", "Bottom row")
    ]

    for row_chars, row_name in rows:
        print(f"\n{row_name}:")
        for i, char in enumerate(row_chars):
            if char in training and char in web:
                x_train, y_train, x01_train, y01_train = training[char]
                x_web, y_web = web[char]

                print(f"  {char}: Training[0,1]=({x01_train:.3f}, {y01_train:.3f}), "
                      f"Web=({x_web:.3f}, {y_web:.3f})")

                # Highlight mismatches
                if abs(x01_train - x_web) > 0.01:
                    print(f"      ⚠️  X mismatch: Training={x01_train:.3f}, Web={x_web:.3f}")
                if abs(y01_train - y_web) > 0.01:
                    print(f"      ⚠️  Y mismatch: Training={y01_train:.3f}, Web={y_web:.3f}")


def compute_correct_web_positions():
    """Compute what the web positions SHOULD be to match training"""
    print("\n" + "="*70)
    print("CORRECT WEB POSITIONS (to match training):")
    print("="*70)

    layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

    print("\nkey_positions = {")
    for row_idx, row in enumerate(layout):
        for col_idx, char in enumerate(row):
            x01 = (col_idx + 0.5) / 10.0
            y01 = (row_idx + 0.5) / 3.0
            print(f"    '{char}': ({x01:.3f}, {y01:.3f}),")
    print("}")


if __name__ == "__main__":
    compare_layouts()
    compute_correct_web_positions()