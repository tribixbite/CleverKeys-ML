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
  await page.goto('http://localhost:8888/test-stateful.html');

  console.log('Waiting 5 seconds for models to load...');
  await page.waitForTimeout(5000);

  // Check button state
  const isDisabled = await page.evaluate(() => {
    const btn = document.querySelector('#testBtn');
    return btn ? btn.disabled : null;
  });

  console.log('Button disabled:', isDisabled);

  // Check status text
  const statusText = await page.evaluate(() => {
    const status = document.querySelector('#status');
    return status ? status.textContent : null;
  });

  console.log('Status text:', statusText);

  await browser.close();
})();