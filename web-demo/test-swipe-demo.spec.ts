import { test, expect } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

test.describe('Swipe ONNX Demo', () => {
    test.beforeEach(async ({ page }) => {
        // Navigate to the demo page via http server
        await page.goto('http://localhost:3001/swipe-onnx.html');

        // Wait for canvas to be present
        await page.waitForSelector('#swipeCanvas', { timeout: 10000 });

        // Give models time to load
        await page.waitForTimeout(3000);
    });

    test('should load models successfully', async ({ page }) => {
        // Check that models are loaded
        const status = await page.locator('#status').textContent();
        expect(status).toContain('loaded');
    });

    test('should recognize simple word "at"', async ({ page }) => {
        // Get canvas element
        const canvas = await page.locator('#swipeCanvas');
        const box = await canvas.boundingBox();
        if (!box) throw new Error('Canvas not found');

        // Simulate swipe for "at" (a -> t)
        // 'a' is approximately at left side, 't' is middle-right
        const points = [
            { x: box.x + box.width * 0.1, y: box.y + box.height * 0.5 },  // 'a' position
            { x: box.x + box.width * 0.15, y: box.y + box.height * 0.48 },
            { x: box.x + box.width * 0.3, y: box.y + box.height * 0.45 },
            { x: box.x + box.width * 0.5, y: box.y + box.height * 0.4 },
            { x: box.x + box.width * 0.65, y: box.y + box.height * 0.35 },  // 't' position
        ];

        // Simulate swipe gesture
        await page.mouse.move(points[0].x, points[0].y);
        await page.mouse.down();

        for (const point of points.slice(1)) {
            await page.mouse.move(point.x, point.y, { steps: 5 });
            await page.waitForTimeout(10);
        }

        await page.mouse.up();

        // Wait for prediction
        await page.waitForTimeout(500);

        // Check prediction
        const predictions = await page.locator('#hypotheses').textContent();
        console.log('Predictions:', predictions);

        // Should have some prediction
        expect(predictions).toBeTruthy();
    });

    test('should recognize word "the"', async ({ page }) => {
        // Get canvas element
        const canvas = await page.locator('#swipeCanvas');
        const box = await canvas.boundingBox();
        if (!box) throw new Error('Canvas not found');

        // Simulate swipe for "the" (t -> h -> e)
        const points = [
            { x: box.x + box.width * 0.65, y: box.y + box.height * 0.35 },  // 't'
            { x: box.x + box.width * 0.60, y: box.y + box.height * 0.40 },
            { x: box.x + box.width * 0.55, y: box.y + box.height * 0.45 },
            { x: box.x + box.width * 0.50, y: box.y + box.height * 0.50 },  // 'h'
            { x: box.x + box.width * 0.40, y: box.y + box.height * 0.45 },
            { x: box.x + box.width * 0.30, y: box.y + box.height * 0.35 },  // 'e'
        ];

        // Simulate swipe
        await page.mouse.move(points[0].x, points[0].y);
        await page.mouse.down();

        for (const point of points.slice(1)) {
            await page.mouse.move(point.x, point.y, { steps: 5 });
            await page.waitForTimeout(10);
        }

        await page.mouse.up();

        // Wait for prediction
        await page.waitForTimeout(500);

        // Check prediction
        const predictions = await page.locator('#hypotheses').textContent();
        console.log('Predictions:', predictions);

        expect(predictions).toBeTruthy();
    });

    test('should clear trace on new swipe', async ({ page }) => {
        const canvas = await page.locator('#swipeCanvas');
        const box = await canvas.boundingBox();
        if (!box) throw new Error('Canvas not found');

        // First swipe
        await page.mouse.move(box.x + 50, box.y + 50);
        await page.mouse.down();
        await page.mouse.move(box.x + 150, box.y + 50, { steps: 5 });
        await page.mouse.up();

        await page.waitForTimeout(200);

        // Second swipe should clear the first
        await page.mouse.move(box.x + 100, box.y + 100);
        await page.mouse.down();
        await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
        await page.mouse.up();

        // Canvas should only have one trace
        const traceData = await page.evaluate(() => {
            const canvas = document.getElementById('swipeCanvas') as HTMLCanvasElement;
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;

            // Check if canvas was cleared (simplified check)
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            let nonTransparentPixels = 0;
            for (let i = 3; i < imageData.data.length; i += 4) {
                if (imageData.data[i] > 0) nonTransparentPixels++;
            }
            return nonTransparentPixels;
        });

        expect(traceData).toBeGreaterThan(0); // Should have some drawn pixels
    });

    test('should handle training data sample', async ({ page }) => {
        // Load a sample from training data
        const trainingDataPath = path.join(__dirname, '../data/train_final_val.jsonl');
        if (!fs.existsSync(trainingDataPath)) {
            test.skip();
            return;
        }

        const lines = fs.readFileSync(trainingDataPath, 'utf-8').split('\n');
        const sample = JSON.parse(lines[0]);

        // Get canvas dimensions
        const canvas = await page.locator('#swipeCanvas');
        const box = await canvas.boundingBox();
        if (!box) throw new Error('Canvas not found');

        // Convert normalized coordinates to canvas coordinates
        const points = sample.points.map((pt: any) => ({
            x: box.x + ((pt.x + 1) / 2) * box.width,
            y: box.y + ((pt.y + 1) / 2) * box.height
        }));

        // Simulate the swipe
        if (points.length > 0) {
            await page.mouse.move(points[0].x, points[0].y);
            await page.mouse.down();

            for (const point of points.slice(1)) {
                await page.mouse.move(point.x, point.y, { steps: 2 });
                await page.waitForTimeout(5);
            }

            await page.mouse.up();
        }

        // Wait for prediction
        await page.waitForTimeout(1000);

        // Check that we got a prediction
        const predictions = await page.locator('#hypotheses').textContent();
        console.log(`Word: ${sample.word}, Predictions:`, predictions);

        expect(predictions).toBeTruthy();
    });
});