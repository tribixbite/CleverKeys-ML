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

  console.log('Navigating to page...');
  await page.goto('http://localhost:8888/test-stateful-fixed.html');

  console.log('Waiting for models to load...');
  await page.waitForSelector('#testBtn:not([disabled])', { timeout: 60000 });

  console.log('Clicking test button...');
  await page.click('#testBtn');

  console.log('Waiting for processing...');
  await page.waitForTimeout(3000);

  // Check result
  const hasResult = await page.evaluate(() => {
    const result = document.querySelector('#result');
    return result && result.innerHTML.trim() !== '';
  });

  console.log('Has result:', hasResult);

  if (hasResult) {
    const resultText = await page.evaluate(() => {
      const result = document.querySelector('#result');
      return result ? result.textContent : null;
    });
    console.log('\nResult:', resultText);
  }

  await browser.close();
})();