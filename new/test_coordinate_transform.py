#!/usr/bin/env python3
"""
Test script to verify coordinate transformation from [0,1] to [-1,1] is working correctly.
"""

import sys
sys.path.append('/home/will/git/swype/cleverkeys/new')

from train_transducer_personalized import PersonalizedSwipeDataset


def test_coordinate_transform():
    """Test that coordinates are correctly transformed from [0,1] to [-1,1]."""

    # Test cases with expected transformations
    test_cases = [
        # Input points in [0,1] range -> Expected output in [-1,1] range
        ([{"x": 0.0, "y": 0.0, "t": 0}], [-1.0, -1.0]),  # Top-left Q key -> (-1,-1)
        ([{"x": 1.0, "y": 1.0, "t": 0}], [1.0, 1.0]),    # Bottom-right -> (1,1)
        ([{"x": 0.5, "y": 0.5, "t": 0}], [0.0, 0.0]),    # Center -> (0,0)
        ([{"x": 0.25, "y": 0.75, "t": 0}], [-0.5, 0.5]), # Quarter positions
    ]

    print("Testing coordinate transformation from [0,1] to [-1,1]...")
    print("=" * 60)

    all_passed = True

    for points, expected in test_cases:
        # Use the static method to prepare points
        prepared = PersonalizedSwipeDataset._prepare_points(points)

        if prepared:
            actual_x = prepared[0]["x"]
            actual_y = prepared[0]["y"]
            expected_x, expected_y = expected

            # Check with small tolerance for floating point comparison
            x_match = abs(actual_x - expected_x) < 0.001
            y_match = abs(actual_y - expected_y) < 0.001

            status = "✓ PASS" if (x_match and y_match) else "✗ FAIL"

            print(f"Input:    ({points[0]['x']:.2f}, {points[0]['y']:.2f})")
            print(f"Expected: ({expected_x:.2f}, {expected_y:.2f})")
            print(f"Actual:   ({actual_x:.2f}, {actual_y:.2f})")
            print(f"Status:   {status}")
            print("-" * 40)

            if not (x_match and y_match):
                all_passed = False

    # Test edge cases with out-of-bounds values
    print("\nTesting edge cases with clamping...")
    edge_cases = [
        ([{"x": -0.1, "y": 0.5, "t": 0}], "Should handle negative x"),
        ([{"x": 1.2, "y": 0.5, "t": 0}], "Should handle x > 1"),
        ([{"x": 0.5, "y": -0.1, "t": 0}], "Should handle negative y"),
        ([{"x": 0.5, "y": 1.2, "t": 0}], "Should handle y > 1"),
    ]

    for points, description in edge_cases:
        prepared = PersonalizedSwipeDataset._prepare_points(points)
        if prepared:
            actual_x = prepared[0]["x"]
            actual_y = prepared[0]["y"]

            # Check that values are clamped to [-1.5, 1.5]
            x_clamped = -1.5 <= actual_x <= 1.5
            y_clamped = -1.5 <= actual_y <= 1.5

            status = "✓ PASS" if (x_clamped and y_clamped) else "✗ FAIL"

            print(f"{description}")
            print(f"Input:    ({points[0]['x']:.2f}, {points[0]['y']:.2f})")
            print(f"Output:   ({actual_x:.2f}, {actual_y:.2f})")
            print(f"Clamped:  {status}")
            print("-" * 40)

            if not (x_clamped and y_clamped):
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Coordinate transformation is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please check the transformation logic!")

    return all_passed


if __name__ == "__main__":
    success = test_coordinate_transform()
    sys.exit(0 if success else 1)