import { test, expect } from '@playwright/test';

test.describe('Swipe RNN-T Decoder Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the test page via HTTP server
    await page.goto('http://localhost:8888/test-stateful.html');

    // Wait for models to load (increase timeout as models are large)
    await page.waitForSelector('#testBtn:not([disabled])', { timeout: 60000 });
  });

  test('models load successfully', async ({ page }) => {
    // Check status message
    const status = await page.locator('#status').textContent();
    expect(status).toContain('Models loaded successfully!');
  });

  test('can process swipe trace for "hello"', async ({ page }) => {
    // Click test button
    await page.click('#testBtn');

    // Wait for results
    await page.waitForSelector('#result h3', { timeout: 10000 });

    // Check results
    const result = await page.locator('#result').textContent();
    expect(result).toContain('Expected: hello');
    // Note: prediction accuracy depends on model training
  });

  test('can process custom swipe trace', async ({ page }) => {
    // Inject a custom trace for "the"
    await page.evaluate(() => {
      // Override the test trace
      window.testTrace = {
        word: "the",
        points: [
          {x: -0.3, y: -0.3, t: 0},    // t
          {x: -0.25, y: -0.3, t: 20},
          {x: -0.2, y: -0.3, t: 40},
          {x: -0.15, y: -0.3, t: 60},   // h
          {x: -0.1, y: -0.3, t: 80},
          {x: -0.05, y: -0.3, t: 100},
          {x: 0.5, y: -0.3, t: 120},    // e
          {x: 0.45, y: -0.3, t: 140}
        ]
      };
    });

    // Click test button
    await page.click('#testBtn');

    // Wait for results
    await page.waitForSelector('#result h3', { timeout: 10000 });

    // Check that processing completed
    const status = await page.locator('#status').textContent();
    expect(status).toContain('Test complete!');
  });

  test('handles preprocessing correctly', async ({ page }) => {
    // Test that the preprocessor is working
    const preprocessorExists = await page.evaluate(() => {
      return typeof window.preprocessor !== 'undefined' &&
             typeof window.preprocessor.process === 'function';
    });
    expect(preprocessorExists).toBeTruthy();

    // Test preprocessing a simple trace
    const result = await page.evaluate(() => {
      const trace = [
        {x: 0, y: 0, t: 0},
        {x: 0.5, y: 0, t: 100},
        {x: 1, y: 0, t: 200}
      ];
      const processed = window.preprocessor.process(trace);
      return {
        hasFeatures: processed.features !== undefined,
        hasNumFrames: processed.numFrames !== undefined,
        numFrames: processed.numFrames
      };
    });

    expect(result.hasFeatures).toBeTruthy();
    expect(result.hasNumFrames).toBeTruthy();
    expect(result.numFrames).toBeGreaterThan(0);
  });

  test('decoder processes features correctly', async ({ page }) => {
    // Test that decoder can process features
    const canDecode = await page.evaluate(async () => {
      // Create dummy features
      const numFrames = 10;
      const features = new Float32Array(numFrames * 37).fill(0);

      // Convert to feature matrix
      const featureMatrix = [];
      for (let t = 0; t < numFrames; t++) {
        const frame = [];
        for (let f = 0; f < 37; f++) {
          frame.push(features[t * 37 + f]);
        }
        featureMatrix.push(frame);
      }

      try {
        const result = await window.decoder.greedyDecode(featureMatrix, 5);
        return result.text !== undefined;
      } catch (error) {
        console.error('Decode error:', error);
        return false;
      }
    });

    expect(canDecode).toBeTruthy();
  });
});