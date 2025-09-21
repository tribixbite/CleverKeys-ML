const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Listen for console messages
  page.on('console', msg => {
    if (msg.type() === 'log') {
      console.log(`Page log: ${msg.text()}`);
    }
  });

  // Listen for errors
  page.on('pageerror', err => {
    console.log(`Page error: ${err.message}`);
  });

  console.log('Navigating to final demo page...');
  await page.goto('http://localhost:8888/swipe-demo-final.html');

  console.log('Waiting for models to load...');
  await page.waitForSelector('#testSingleBtn:not([disabled])', { timeout: 60000 });

  // Test single word
  console.log('\n=== Testing Single Word ===');
  await page.click('#testSingleBtn');
  await page.waitForTimeout(2000);

  const singleResult = await page.evaluate(() => {
    const result = document.querySelector('#results .result');
    return result ? result.textContent : null;
  });
  console.log('Single result:', singleResult);

  // Test multiple words
  console.log('\n=== Testing Multiple Words ===');
  await page.click('#testMultipleBtn');
  await page.waitForTimeout(3000);

  const multipleResult = await page.evaluate(() => {
    const accuracy = document.querySelector('#results h4');
    return accuracy ? accuracy.textContent : null;
  });
  console.log('Multiple result:', multipleResult);

  // Test beam search
  console.log('\n=== Testing Beam Search ===');
  await page.click('#testBeamBtn');
  await page.waitForTimeout(3000);

  const beamResult = await page.evaluate(() => {
    const results = document.querySelectorAll('#results .result');
    const lastResult = results[results.length - 1];
    return lastResult ? lastResult.textContent : null;
  });
  console.log('Beam search result:', beamResult);

  await browser.close();
})();