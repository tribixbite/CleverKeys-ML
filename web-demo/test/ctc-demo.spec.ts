import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Assumes a static server is serving the web-demo directory root, e.g.:
//   npx http-server web-demo -p 8787
// or
//   python -m http.server 8787 --directory web-demo

const BASE = process.env.BASE_URL || 'http://127.0.0.1:8787';

function readDatasetWord(word: string) {
  const ds = path.resolve(__dirname, '../../data/train_final_train.jsonl');
  const stream = fs.createReadStream(ds, { encoding: 'utf-8' });
  const rl = require('readline').createInterface({ input: stream });
  let found: any = null;
  return new Promise<any>((resolve) => {
    rl.on('line', (line: string) => {
      if (found) return;
      try {
        const rec = JSON.parse(line);
        if (rec && typeof rec.word === 'string' && rec.word.toLowerCase() === word) {
          found = rec;
          rl.close();
        }
      } catch {}
    });
    rl.on('close', () => resolve(found));
    rl.on('error', () => resolve(null));
  });
}

test.describe('CTC Demo (Squeezeformer-CTC)', () => {
  test('predicts hello from dataset trace', async ({ page }) => {
    await page.goto(`${BASE}/swipe-onnx-modular.html`);
    await page.getByText('Ready').first().waitFor({ state: 'visible', timeout: 30000 });
    await page.locator('#modelType').selectOption('ctc');

    const rec: any = await readDatasetWord('hello');
    expect(rec).toBeTruthy();

    const out = await page.evaluate(async (pts) => {
      // @ts-ignore
      const r = await (window as any).ctcDecoder.simpleCTCDecode(pts);
      return r.text;
    }, rec.points);

    expect(out).toBe('hello');
  });

  test('predicts person from dataset trace', async ({ page }) => {
    await page.goto(`${BASE}/swipe-onnx-modular.html`);
    await page.getByText('Ready').first().waitFor({ state: 'visible', timeout: 30000 });
    await page.locator('#modelType').selectOption('ctc');

    const rec: any = await readDatasetWord('person');
    expect(rec).toBeTruthy();

    const out = await page.evaluate(async (pts) => {
      // @ts-ignore
      const r = await (window as any).ctcDecoder.simpleCTCDecode(pts);
      return r.text;
    }, rec.points);

    expect(out).toBe('person');
  });
});
