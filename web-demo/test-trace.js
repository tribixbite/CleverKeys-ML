const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Listen for console messages
  page.on('console', msg => {
    console.log(`Console ${msg.type()}: ${msg.text()}`);
  });

  console.log('Navigating to page...');
  await page.goto('http://localhost:8888/test-stateful.html');

  // Wait for models to load
  await page.waitForSelector('#testBtn:not([disabled])', { timeout: 60000 });

  // Check the test trace and processed features
  const traceInfo = await page.evaluate(() => {
    const trace = window.testTrace || {
      word: "hello",
      points: [
        {x: -0.15, y: 0.0, t: 0},     // h
        {x: -0.1, y: 0.0, t: 20},
        {x: 0.5, y: -0.3, t: 40},      // e
        {x: 0.45, y: -0.35, t: 60},
        {x: 0.2, y: -0.3, t: 80},      // l
        {x: 0.15, y: -0.35, t: 100},
        {x: 0.2, y: -0.3, t: 120},     // l
        {x: 0.15, y: -0.35, t: 140},
        {x: 0.6, y: -0.3, t: 160},     // o
        {x: 0.55, y: -0.35, t: 180}
      ]
    };

    // Process the trace
    const preprocessor = new SwipePreprocessor();
    const processed = preprocessor.process(trace.points);

    // Get keyboard layout from feature extractor
    const layout = preprocessor.extractor.keyCenters;

    return {
      trace: trace,
      processed: {
        numFrames: processed.numFrames,
        duration: processed.duration,
        firstFeatures: Array.from(processed.features.slice(0, 37))
      },
      keyboardLayout: layout.slice(0, 10) // First 10 keys
    };
  });

  console.log('\n=== Trace Analysis ===');
  console.log('Word:', traceInfo.trace.word);
  console.log('Points:', JSON.stringify(traceInfo.trace.points, null, 2));
  console.log('\nProcessed:');
  console.log('  Frames:', traceInfo.processed.numFrames);
  console.log('  Duration:', traceInfo.processed.duration);
  console.log('  First frame features (x, y, t, ...):',
    traceInfo.processed.firstFeatures.slice(0, 5).map(f => f.toFixed(3)).join(', '));

  console.log('\nKeyboard layout (first row):');
  traceInfo.keyboardLayout.forEach(key => {
    console.log(`  ${key.char}: x=${key.x.toFixed(3)}, y=${key.y.toFixed(3)}`);
  });

  await browser.close();
})();