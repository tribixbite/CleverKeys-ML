#!/usr/bin/env python3
"""Test the web demo using Playwright to verify appearance and functionality."""

import asyncio
from playwright.async_api import async_playwright
import time

async def test_web_demo():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()

        print("Navigating to web demo...")
        await page.goto("http://localhost:8081/swipe-onnx-modular.html")
        await page.wait_for_load_state('networkidle')

        # Wait for models to load
        print("Waiting for models to load...")
        await page.wait_for_timeout(3000)

        # Take a screenshot for visual verification
        await page.screenshot(path="demo_screenshot.png")
        print("Screenshot saved as demo_screenshot.png")

        # Check status text
        status = await page.text_content("#status")
        print(f"Status: {status}")

        # Test 1: Check aesthetic elements
        print("\n=== Testing Aesthetic Elements ===")

        # Check background color
        body_style = await page.evaluate("getComputedStyle(document.body).backgroundColor")
        print(f"Body background: {body_style}")

        # Check if dark theme is applied (gray-900)
        has_dark_bg = await page.evaluate("document.body.classList.contains('bg-gray-900')")
        print(f"Has dark background class: {has_dark_bg}")

        # Check keyboard presence
        keyboard_keys = await page.query_selector_all(".key")
        print(f"Number of keyboard keys: {len(keyboard_keys)}")

        # Check model selector
        model_selector = await page.query_selector("#modelType")
        if model_selector:
            options = await page.evaluate("""
                Array.from(document.querySelector('#modelType').options).map(o => o.text)
            """)
            print(f"Model options: {options}")

        # Test 2: Single letter typing
        print("\n=== Testing Single Letter Typing ===")

        # Click on letter 'H'
        await page.click('[data-key="h"]')
        await page.wait_for_timeout(500)

        # Click on letter 'I'
        await page.click('[data-key="i"]')
        await page.wait_for_timeout(500)

        # Check input text
        input_text = await page.text_content("#inputText")
        print(f"Input text after typing 'hi': '{input_text}'")

        # Test 3: Special keys
        print("\n=== Testing Special Keys ===")

        # Test space key
        space_key = await page.query_selector('[data-key=" "]')
        if space_key:
            await space_key.click()
            await page.wait_for_timeout(500)
            input_text = await page.text_content("#inputText")
            print(f"After space: '{input_text}'")

        # Test backspace
        backspace_key = await page.query_selector('[data-key="backspace"]')
        if backspace_key:
            await backspace_key.click()
            await page.wait_for_timeout(500)
            input_text = await page.text_content("#inputText")
            print(f"After backspace: '{input_text}'")

        # Test 4: Clear button
        print("\n=== Testing Clear Button ===")
        await page.click("#clearBtn")
        await page.wait_for_timeout(500)
        input_text = await page.text_content("#inputText")
        print(f"After clear: '{input_text}'")

        # Test 5: Swipe gesture (simulate swipe for "hello")
        print("\n=== Testing Swipe Gesture ===")

        # Get keyboard bounds
        keyboard_elem = await page.query_selector(".keyboard-rows")
        if keyboard_elem:
            keyboard_box = await keyboard_elem.bounding_box()
            print(f"Keyboard bounds: {keyboard_box}")

            # Simulate swipe for "hello" - h -> e -> l -> l -> o
            # This would need the actual key positions
            # For now, just test that canvas exists
            canvas = await page.query_selector("#swipeCanvas")
            if canvas:
                print("Swipe canvas found")
                canvas_box = await canvas.bounding_box()
                print(f"Canvas bounds: {canvas_box}")

                # Try the test button instead
                await page.click("#testBtn")
                await page.wait_for_timeout(2000)

                # Check if prediction appeared
                suggestions = await page.query_selector_all("#suggestions button")
                if suggestions:
                    print(f"Suggestions found: {len(suggestions)}")
                    for i, suggestion in enumerate(suggestions[:3]):
                        text = await suggestion.text_content()
                        print(f"  Suggestion {i+1}: {text}")

        # Test 6: Model switching
        print("\n=== Testing Model Switching ===")

        # Switch to CTC model
        await page.select_option("#modelType", "ctc")
        await page.wait_for_timeout(2000)
        status = await page.text_content("#status")
        print(f"Status after switching to CTC: {status}")

        # Switch back to RNN-T
        await page.select_option("#modelType", "rnnt")
        await page.wait_for_timeout(2000)
        status = await page.text_content("#status")
        print(f"Status after switching back to RNN-T: {status}")

        # Test 7: Debug mode
        print("\n=== Testing Debug Mode ===")
        await page.click("#debugBtn")
        await page.wait_for_timeout(500)
        debug_btn_text = await page.text_content("#debugBtn")
        print(f"Debug button text: {debug_btn_text}")

        # Check if debug output is visible
        debug_output = await page.query_selector("#debugOutput")
        if debug_output:
            is_visible = await debug_output.is_visible()
            print(f"Debug output visible: {is_visible}")

        print("\n=== Test Complete ===")

        # Keep browser open for manual inspection
        print("Browser will stay open for 10 seconds for manual inspection...")
        await page.wait_for_timeout(10000)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_web_demo())