import { test, expect } from '@playwright/test';

test.describe('Modular ONNX Demo (best_latest)', () => {
  test('loads and initializes RNN-T models', async ({ page }) => {
    await page.goto('http://localhost:3001/swipe-onnx-modular.html');
    await page.waitForSelector('#statusText', { timeout: 60000 });
    const status = await page.locator('#statusText').textContent();
    expect(status || '').toContain('Ready - RNN-T model loaded');
  });
});

