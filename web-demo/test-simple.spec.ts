import { test, expect } from '@playwright/test';

test('basic page load', async ({ page }) => {
    // Navigate to the demo page
    await page.goto('http://localhost:3001/swipe-onnx.html');

    // Check title
    await expect(page).toHaveTitle(/Neural Swipe/);

    // Check canvas exists
    const canvas = await page.locator('#swipeCanvas');
    await expect(canvas).toBeVisible();

    console.log('Page loaded successfully');
});