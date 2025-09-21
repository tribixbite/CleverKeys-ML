const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Listen for console messages
  page.on('console', msg => {
    console.log(`Console ${msg.type()}: ${msg.text()}`);
  });

  // Listen for errors
  page.on('pageerror', err => {
    console.log(`Page error: ${err.message}`);
  });

  console.log('Navigating to CTC test page...');
  await page.goto('http://localhost:8888/test-ctc.html');

  console.log('Waiting for models to load...');
  await page.waitForSelector('#testBtn:not([disabled])', { timeout: 60000 });

  // Test autoregressive decoding
  console.log('\n=== Testing CTC Autoregressive ===');
  await page.click('#testBtn');
  await page.waitForTimeout(3000);

  const autoregressiveResult = await page.evaluate(() => {
    const result = document.querySelector('#result');
    return result ? result.textContent : null;
  });
  console.log('Result:', autoregressiveResult);

  // Test simple CTC
  console.log('\n=== Testing Simple CTC ===');
  await page.click('#testSimpleBtn');
  await page.waitForTimeout(2000);

  const simpleResult = await page.evaluate(() => {
    const result = document.querySelector('#result');
    return result ? result.textContent : null;
  });
  console.log('Result:', simpleResult);

  await browser.close();
})();