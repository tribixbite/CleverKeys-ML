#!/usr/bin/env python3
"""Test the swipe demo with model switching using Playwright."""

from playwright.sync_api import sync_playwright
import time


def test_demo():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)  # Set to True for headless
        context = browser.new_context()
        page = context.new_page()

        # Navigate to demo
        print("Opening demo at http://localhost:8080/swipe-onnx.html")
        page.goto("http://localhost:8080/swipe-onnx.html")

        # Wait for page to load
        time.sleep(3)

        # Take initial screenshot
        page.screenshot(path="demo_initial.png")
        print("✓ Demo loaded, screenshot saved as demo_initial.png")

        # Check if model selector exists
        selector = page.query_selector("#modelSelector")
        if selector:
            print("✓ Model selector dropdown found")

            # Get current model
            current_model = selector.evaluate("el => el.value")
            print(f"  Current model: {current_model}")

            # Get all options
            options = selector.evaluate("""el => {
                return Array.from(el.options).map(opt => ({
                    value: opt.value,
                    text: opt.text
                }));
            }""")
            print("  Available models:")
            for opt in options:
                print(f"    - {opt['text']} ({opt['value']})")

            # Switch to Android INT8 model
            print("\nSwitching to Android INT8 model...")
            selector.select_option("encoder_android_int8_final.onnx")
            time.sleep(2)

            # Check status
            status = page.query_selector("#modelStatus")
            if status:
                status_text = status.text_content()
                print(f"  Model status: {status_text}")

        # Test gesture input
        print("\n\nTesting gesture input...")
        canvas = page.query_selector("#swipeCanvas")
        if canvas:
            print("✓ Canvas found")

            # Get canvas bounds
            box = canvas.bounding_box()
            if box:
                # Simulate swipe for "hello"
                # h -> e -> l -> l -> o
                points = [
                    (box['x'] + box['width'] * 0.6, box['y'] + box['height'] * 0.5),  # h
                    (box['x'] + box['width'] * 0.3, box['y'] + box['height'] * 0.25),  # e
                    (box['x'] + box['width'] * 0.85, box['y'] + box['height'] * 0.5),  # l
                    (box['x'] + box['width'] * 0.85, box['y'] + box['height'] * 0.5),  # l (same)
                    (box['x'] + box['width'] * 0.8, box['y'] + box['height'] * 0.25),  # o
                ]

                # Start swipe
                page.mouse.move(points[0][0], points[0][1])
                page.mouse.down()

                # Move through points
                for x, y in points[1:]:
                    page.mouse.move(x, y)
                    time.sleep(0.05)  # Small delay between points

                # End swipe
                page.mouse.up()
                print("✓ Simulated swipe gesture for 'hello'")

                # Wait for processing
                time.sleep(2)

                # Take screenshot after swipe
                page.screenshot(path="demo_after_swipe.png")
                print("✓ Screenshot saved as demo_after_swipe.png")

                # Check debug info
                debug_overlay = page.query_selector(".debug-overlay")
                if debug_overlay:
                    debug_text = debug_overlay.text_content()
                    if debug_text:
                        print(f"\nDebug info: {debug_text[:200]}")

        # Test model switching multiple times
        print("\n\nTesting rapid model switching...")
        models = ["encoder_fp32.onnx", "encoder_web_ultra.onnx", "encoder_android_int8_final.onnx"]

        for model in models:
            print(f"  Switching to {model}...")
            selector.select_option(model)
            time.sleep(1.5)

            status = page.query_selector("#modelStatus")
            if status:
                print(f"    Status: {status.text_content()}")

        print("\n✅ All tests completed successfully!")

        # Keep browser open for manual inspection
        input("Press Enter to close browser...")
        browser.close()


if __name__ == "__main__":
    test_demo()