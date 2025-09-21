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

  console.log('Navigating to modular test page...');
  await page.goto('http://localhost:8888/swipe-onnx-modular.html');

  console.log('Waiting for models to load...');

  // Wait for status to show "Ready"
  await page.waitForFunction(
    () => {
      const status = document.getElementById('status');
      return status && status.textContent.includes('Ready');
    },
    { timeout: 60000 }
  );

  console.log('Models loaded successfully!');

  // Test RNN-T model
  console.log('\n=== Testing RNN-T Model ===');
  await page.selectOption('#modelType', 'rnnt');
  await page.click('#testBtn');
  await page.waitForTimeout(3000);

  const rnntResult = await page.evaluate(() => {
    const suggestions = document.querySelector('#suggestions');
    const inputText = document.querySelector('#inputText');
    return {
      suggestions: suggestions ? suggestions.textContent : null,
      inputText: inputText ? inputText.textContent : null
    };
  });
  console.log('RNN-T Result:', rnntResult);

  // Test CTC model
  console.log('\n=== Testing CTC Model ===');
  await page.selectOption('#modelType', 'ctc');
  await page.click('#testBtn');
  await page.waitForTimeout(3000);

  const ctcResult = await page.evaluate(() => {
    const suggestions = document.querySelector('#suggestions');
    const inputText = document.querySelector('#inputText');
    return {
      suggestions: suggestions ? suggestions.textContent : null,
      inputText: inputText ? inputText.textContent : null
    };
  });
  console.log('CTC Result:', ctcResult);

  // Check for any errors
  const errors = await page.evaluate(() => {
    const status = document.getElementById('status');
    return status && status.className.includes('error') ? status.textContent : null;
  });

  if (errors) {
    console.error('Errors found:', errors);
  } else {
    console.log('\n✅ Both models tested successfully!');
  }

  await browser.close();
})();