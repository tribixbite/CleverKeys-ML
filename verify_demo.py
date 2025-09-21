#!/usr/bin/env python3
"""Quick verification script for the web demo using Playwright"""

import asyncio
from playwright.async_api import async_playwright
import time

async def test_swipe_demo():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Set up console message listener
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))

        # Navigate to the demo
        print("🌐 Loading demo page...")
        await page.goto("http://localhost:3001/swipe-onnx.html")

        # Wait for models to load
        print("⏳ Waiting for models to load...")
        await page.wait_for_timeout(5000)

        # Check for critical errors
        errors = [msg for msg in console_messages if "error" in msg.lower() or "❌" in msg]

        if errors:
            print("\n❌ Found errors in console:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")
        else:
            print("✅ No critical errors found in console")

        # Check if models loaded successfully
        model_loaded = await page.evaluate("""
            () => {
                return {
                    encoder: typeof encoderSession !== 'undefined' && encoderSession !== null,
                    decoder: typeof decoderSession !== 'undefined' && decoderSession !== null,
                    ready: typeof isModelReady !== 'undefined' && isModelReady
                }
            }
        """)

        print(f"\n📊 Model status:")
        print(f"  Encoder loaded: {model_loaded.get('encoder', False)}")
        print(f"  Decoder loaded: {model_loaded.get('decoder', False)}")
        print(f"  System ready: {model_loaded.get('ready', False)}")

        # Try a simple swipe gesture
        print("\n🎯 Testing swipe gesture...")
        canvas = await page.query_selector("#swipeCanvas")

        if canvas:
            # Simulate a swipe for the word "the"
            await canvas.hover()
            await page.mouse.down()

            # Move through t-h-e positions
            positions = [
                (200, 100),  # t
                (250, 100),  # h
                (300, 100),  # e
            ]

            for x, y in positions:
                await page.mouse.move(x, y)
                await page.wait_for_timeout(50)

            await page.mouse.up()
            await page.wait_for_timeout(2000)

            # Check for inference results
            recent_messages = console_messages[-20:]
            inference_success = any("inference completed" in msg.lower() or "✓ Encoder output" in msg for msg in recent_messages)

            if inference_success:
                print("✅ Inference completed successfully")
                # Show recent relevant messages
                for msg in recent_messages:
                    if any(keyword in msg.lower() for keyword in ["encoder output", "inference completed", "prediction", "frames"]):
                        print(f"  {msg}")
            else:
                print("❌ Inference did not complete")
                # Check for specific error about input names
                input_errors = [msg for msg in recent_messages if "audio_signal" in msg or "missing in 'feeds'" in msg]
                if input_errors:
                    print("  ⚠️  Still has input name mismatch errors:")
                    for err in input_errors[:2]:
                        print(f"    {err}")

        await browser.close()

        # Final verdict
        print("\n" + "="*50)
        if not errors and model_loaded.get('ready', False):
            print("✅ DEMO VERIFICATION PASSED")
            return True
        else:
            print("❌ DEMO VERIFICATION FAILED - Issues detected")
            return False

if __name__ == "__main__":
    result = asyncio.run(test_swipe_demo())
    exit(0 if result else 1)