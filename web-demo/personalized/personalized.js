const CONFIG = {
  maxSymbols: 15,
  defaultBlankId: 0,
  resample: {
    shortTarget: 56,
    longTarget: 96,
    shortThreshold: 48,
    longThreshold: 112,
  },
};

const MIN_DRAG_DISTANCE = 0.05; // normalized units (~5% of keyboard width)
const MIN_DRAG_TIME = 80;       // ms before we promote a tap into a swipe
const DEBUG_HISTORY_LIMIT = 12;

const KEY_LAYOUT = [
  "qwertyuiop",
  "asdfghjkl",
  "zxcvbnm",
];

const KEY_CENTERS = (() => {
  const centers = [];
  KEY_LAYOUT.forEach((row, r) => {
    [...row].forEach((ch, c) => {
      const x01 = (c + 0.5) / 10.0;
      const y01 = (r + 0.5) / 3.0;
      centers.push([ch, x01 * 2 - 1, y01 * 2 - 1]);
    });
  });
  return centers;
})();

const DEBUG = {
  enabled: false,
  current: null,
  history: [],
};

class PersonalizedFeaturizer {
  constructor(cfg = CONFIG.resample) {
    this.cfg = cfg;
  }

  extract(points) {
    if (!points.length) {
      return { features: new Float32Array(37), length: 1 };
    }
    const norm = this._normalize(points);
    const target = this._targetLength(norm.length);
    const processed = this._resample(norm, target);
    const feats = new Float32Array(processed.length * 37);
    for (let i = 0; i < processed.length; i++) {
      feats.set(this._featureVector(processed, i), i * 37);
    }
    return { features: feats, length: processed.length, processed };
  }

  _normalize(points) {
    const startT = points[0].t ?? 0;
    return points.map((pt, idx) => {
      const rawX = typeof pt.x === 'number' ? pt.x : 0.5;
      const rawY = typeof pt.y === 'number' ? pt.y : 0.5;
      const centeredX = Math.max(-1, Math.min(1, rawX * 2 - 1));
      const centeredY = Math.max(-1, Math.min(1, rawY * 2 - 1));
      const t = (typeof pt.t === 'number' ? pt.t : idx * 10) - startT;
      return { x: centeredX, y: centeredY, t: Math.max(0, t) };
    });
  }

  _targetLength(length) {
    if (length <= 1) return Math.max(length, this.cfg.shortTarget);
    if (length <= this.cfg.shortThreshold) return Math.max(length, this.cfg.shortTarget);
    if (length >= this.cfg.longThreshold) return this.cfg.longTarget;
    return length;
  }

  _resample(points, target) {
    if (!points.length || target <= 0) return [];
    if (points.length === target) return points.slice();
    const firstT = points[0].t;
    const lastT = points[points.length - 1].t;
    const duration = Math.max(lastT - firstT, 1);
    const step = duration / Math.max(target - 1, 1);
    let idx = 0;
    const out = [];
    for (let i = 0; i < target; i++) {
      const targetT = i === target - 1 ? lastT : firstT + step * i;
      while (idx < points.length - 2 && points[idx + 1].t < targetT) idx += 1;
      const p1 = points[idx];
      const p2 = points[Math.min(idx + 1, points.length - 1)];
      const span = Math.max(p2.t - p1.t, 1);
      const alpha = Math.max(0, Math.min(1, (targetT - p1.t) / span));
      const x = p1.x + (p2.x - p1.x) * alpha;
      const y = p1.y + (p2.y - p1.y) * alpha;
      out.push({ x, y, t: targetT });
    }
    return out;
  }

  _featureVector(points, idx) {
    const total = points.length;
    const curr = points[idx];
    const prev = idx > 0 ? points[idx - 1] : null;
    const prev2 = idx > 1 ? points[idx - 2] : null;

    const x = curr.x;
    const y = curr.y;
    const tSec = curr.t / 1000;

    let vx = 0, vy = 0, speed = 0;
    if (prev) {
      const dt = Math.max((curr.t - prev.t) / 1000, 0.001);
      vx = (x - prev.x) / dt;
      vy = (y - prev.y) / dt;
      speed = Math.hypot(vx, vy);
    }

    let ax = 0, ay = 0, acc = 0;
    if (prev && prev2) {
      const dt1 = Math.max((curr.t - prev.t) / 1000, 0.001);
      const dt2 = Math.max((prev.t - prev2.t) / 1000, 0.001);
      const vxPrev = (prev.x - prev2.x) / dt2;
      const vyPrev = (prev.y - prev2.y) / dt2;
      ax = (vx - vxPrev) / dt1;
      ay = (vy - vyPrev) / dt1;
      acc = Math.hypot(ax, ay);
    }

    const angle = prev ? Math.atan2(vy, vx) : 0;
    const angleSin = Math.sin(angle);
    const angleCos = Math.cos(angle);

    let curvature = 0;
    if (prev && prev2) {
      const prevAngle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
      curvature = angle - prevAngle;
      while (curvature > Math.PI) curvature -= 2 * Math.PI;
      while (curvature < -Math.PI) curvature += 2 * Math.PI;
    }

    const distances = KEY_CENTERS.map(([, kx, ky]) => Math.hypot(x - kx, y - ky)).sort((a, b) => a - b);
    while (distances.length < 5) distances.push(1.0);

    const progress = idx / Math.max(total - 1, 1);
    const isStart = idx === 0 ? 1 : 0;
    const isEnd = idx === total - 1 ? 1 : 0;

    const window = points.slice(Math.max(0, idx - 2), Math.min(total, idx + 3));
    let meanX = x, stdX = 0, meanY = y, stdY = 0, rangeX = 0, rangeY = 0;
    if (window.length > 1) {
      const xs = window.map(p => p.x);
      const ys = window.map(p => p.y);
      meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
      meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
      stdX = Math.sqrt(xs.reduce((s, v) => s + (v - meanX) ** 2, 0) / xs.length);
      stdY = Math.sqrt(ys.reduce((s, v) => s + (v - meanY) ** 2, 0) / ys.length);
      rangeX = Math.max(...xs) - Math.min(...xs);
      rangeY = Math.max(...ys) - Math.min(...ys);
    }

    const vec = [
      x, y, tSec,
      vx, vy, speed,
      ax, ay, acc,
      angle, angleSin, angleCos,
      curvature,
      ...distances.slice(0, 5),
      progress,
      isStart, isEnd,
      meanX, stdX,
      meanY, stdY,
      rangeX, rangeY,
    ];
    while (vec.length < 37) vec.push(0);
    return new Float32Array(vec.slice(0, 37));
  }
}

class PersonalizedRuntime {
  constructor() {
    this.encoder = null;
    this.step = null;
    this.blankId = CONFIG.defaultBlankId;
    this.charToId = new Map();
    this.idToChar = [];
    this.lexicon = [];
    this.lexiconSet = new Set();
    this.lexiconByLength = new Map();
    this.featurizer = new PersonalizedFeaturizer();
    this.vocabSize = 0;
    this.lstmLayers = 2;
    this.hiddenSize = 320;
  }

  async load() {
    await this._loadRuntimeMeta();
    await this._loadCharVocab();
    await this._loadLexicon();
    await this._loadModels();
  }

  async _loadRuntimeMeta() {
    try {
      const res = await fetch('runtime_meta.json');
      if (!res.ok) return;
      const meta = await res.json();
      this.blankId = typeof meta.blank_id === 'number' ? meta.blank_id : this.blankId;
      if (meta.char_to_id) {
        this.charToId = new Map(Object.entries(meta.char_to_id));
        this.idToChar = Object.entries(meta.id_to_char || {}).reduce((acc, [k, v]) => {
          acc[Number(k)] = v;
          return acc;
        }, []);
      }
    } catch (err) {
      console.warn('runtime_meta.json not loaded', err);
    }
  }

  async _loadCharVocab() {
    if (this.charToId.size && this.idToChar.length) return;
    const res = await fetch('char_vocab.txt');
    if (!res.ok) throw new Error('Failed to load char vocab');
    const lines = (await res.text()).split(/\r?\n/).filter(Boolean);
    this.charToId.clear();
    this.idToChar = [];
    lines.forEach((line, idx) => {
      this.charToId.set(line, idx);
      this.idToChar[idx] = line;
    });
    if (this.blankId >= this.idToChar.length) this.blankId = CONFIG.defaultBlankId;
  }

  async _loadLexicon() {
    try {
      const res = await fetch('words.txt');
      if (!res.ok) throw new Error('missing lexicon');
      const words = (await res.text()).split(/\r?\n/).filter(Boolean);
      this.lexicon = words;
      this.lexiconSet = new Set(words);
      this.lexiconByLength.clear();
      words.forEach((word) => {
        const len = word.length;
        if (!this.lexiconByLength.has(len)) this.lexiconByLength.set(len, []);
        this.lexiconByLength.get(len).push(word);
      });
    } catch (err) {
      console.warn('Lexicon load failed, suggestions disabled', err);
      this.lexicon = [];
       this.lexiconSet = new Set();
      this.lexiconByLength.clear();
    }
  }

  async _loadModels() {
    const provider = (ort.env.webgpu && ort.env.webgpu.enabled) ? 'webgpu' : 'wasm';
    this.encoder = await ort.InferenceSession.create('encoder_int8_qdq.onnx', {
      executionProviders: [provider],
      graphOptimizationLevel: 'all',
    });
    this.step = await ort.InferenceSession.create('rnnt_step_fp32.onnx', {
      executionProviders: [provider],
      graphOptimizationLevel: 'all',
    });

    const stepInputs = this.step.inputNames;
    const inputMeta = this.step.inputMetadata || {};
    const hMeta = inputMeta['h0'];
    if (hMeta?.dimensions?.length === 3) {
      const [layers, , hidden] = hMeta.dimensions;
      if (typeof layers === 'number' && layers > 0) this.lstmLayers = layers;
      if (typeof hidden === 'number' && hidden > 0) this.hiddenSize = hidden;
    }

    const logitsMeta = this.step.outputMetadata?.logits;
    if (logitsMeta?.dimensions?.length === 4) {
      const V = logitsMeta.dimensions[3];
      if (typeof V === 'number' && V > 0) this.vocabSize = V;
    }
    if (!this.vocabSize) this.vocabSize = this.idToChar.length || 30;
    // Many RNNT exports place the blank token at the last index; prefer that.
    this.blankId = this.vocabSize - 1;
  }

  _extractFrames(encodedTensor, length) {
    const dims = encodedTensor.dims;
    const data = encodedTensor.data;
    let frames;
    if (dims[1] === length) {
      const D = dims[2];
      frames = new Float32Array(length * D);
      for (let t = 0; t < length; t++) {
        frames.set(data.slice(t * D, t * D + D), t * D);
      }
      return { frames, D };
    }
    if (dims[2] === length) {
      const D = dims[1];
      frames = new Float32Array(length * D);
      for (let d = 0; d < D; d++) {
        for (let t = 0; t < length; t++) {
          frames[t * D + d] = data[d * length + t];
        }
      }
      return { frames, D };
    }
    const D = dims[2];
    frames = new Float32Array(length * D);
    for (let t = 0; t < length; t++) {
      frames.set(data.slice(t * D, t * D + D), t * D);
    }
    return { frames, D };
  }

  async inferSwipe(points) {
    if (!this.encoder || !this.step) throw new Error('Models not loaded');

    const t0 = performance.now();
    const { features, length, processed } = this.featurizer.extract(points);
    const featurizeMs = performance.now() - t0;

    const encoderInputs = {
      features_bft: new ort.Tensor('float32', features, [1, 37, length]),
      lengths: new ort.Tensor('int32', new Int32Array([length]), [1]),
    };

    const tEnc = performance.now();
    const encOut = await this.encoder.run(encoderInputs);
    const encodeMs = performance.now() - tEnc;

    const encoded = encOut['encoded_btf'];
    const encodedLens = encOut['encoded_lengths'];
    let frameCount = encodedLens && encodedLens.data.length
      ? Number(encodedLens.data[0])
      : (encoded.dims[1] || encoded.dims[2] || length);
    frameCount = Math.max(1, frameCount);
    const { frames, D } = this._extractFrames(encoded, frameCount);

    let h = new ort.Tensor('float32', new Float32Array(this.lstmLayers * this.hiddenSize), [this.lstmLayers, 1, this.hiddenSize]);
    let c = new ort.Tensor('float32', new Float32Array(this.lstmLayers * this.hiddenSize), [this.lstmLayers, 1, this.hiddenSize]);
    let yPrev = new ort.Tensor('int64', new BigInt64Array([BigInt(this.blankId)]), [1]);

    const tDec = performance.now();
    const tokens = [];
    let stepCalls = 0;

    for (let t = 0; t < frameCount; t++) {
      const encFrame = frames.slice(t * D, t * D + D);
      const encTensor = new ort.Tensor('float32', encFrame, [1, D]);

      for (let s = 0; s < CONFIG.maxSymbols; s++) {
        const out = await this.step.run({ y_prev: yPrev, h0: h, c0: c, enc_t: encTensor });
        stepCalls += 1;
        const logits = out.logits.data;
        let bestIdx = 0;
        let bestVal = -Infinity;
        for (let v = 0; v < this.vocabSize; v++) {
          const val = logits[v];
          if (val > bestVal) {
            bestVal = val;
            bestIdx = v;
          }
        }
        h = out.h1;
        c = out.c1;
        yPrev = new ort.Tensor('int64', new BigInt64Array([BigInt(bestIdx)]), [1]);
        if (bestIdx === this.blankId) {
          break;
        }
        tokens.push(bestIdx);
      }
    }

    const decodeMs = performance.now() - tDec;
    const totalMs = performance.now() - t0;

    const word = this._tokensToWord(tokens);
    const suggestions = this._rankSuggestions(word);

    return {
      word,
      suggestions,
      metrics: {
        rawPoints: points.length,
        resampledPoints: processed.length,
        frames: frameCount,
        featuresMs: featurizeMs,
        encodeMs,
        decodeMs,
        totalMs,
        tokens: tokens.length,
        stepCalls,
      },
    };
  }

  _tokensToWord(tokens) {
    const chars = [];
    tokens.forEach((id) => {
      if (id === this.blankId) return;
      const ch = this.idToChar[id];
      if (!ch || ch.startsWith('<')) return;
      chars.push(ch);
    });
    return chars.join('');
  }

  _rankSuggestions(word) {
    if (!this.lexicon.length) return [];
    const target = word.toLowerCase();
    if (!target) return [];
    const direct = this.lexiconSet && this.lexiconSet.has(target) ? [target] : [];
    const pool = new Set(direct);
    const collect = (len) => {
      const bucket = this.lexiconByLength.get(len);
      if (!bucket) return;
      bucket.forEach((candidate) => {
        if (!pool.has(candidate)) pool.add(candidate);
      });
    };
    collect(target.length);
    collect(target.length - 1);
    collect(target.length + 1);
    const ranked = [...pool].map((w) => ({ word: w, score: -this._levenshtein(w, target) }))
      .sort((a, b) => b.score - a.score)
      .map((entry) => entry.word);
    return ranked.slice(0, 3);
  }

  _levenshtein(a, b) {
    const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
    for (let i = 0; i <= a.length; i++) dp[i][0] = i;
    for (let j = 0; j <= b.length; j++) dp[0][j] = j;
    for (let i = 1; i <= a.length; i++) {
      for (let j = 1; j <= b.length; j++) {
        if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1];
        else dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
    return dp[a.length][b.length];
  }
}

const runtime = new PersonalizedRuntime();
let swipePoints = [];
let swipeActive = false;
let pointerStart = null;

function canvasToNormalized(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const normX = (x / rect.width) * 2 - 1;
  const normY = (y / rect.height) * 2 - 1;
  return { normX, normY };
}

function setupCanvas() {
  const canvas = document.getElementById('swipeCanvas');
  const ctx = canvas.getContext('2d');

  const drawPath = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (swipePoints.length < 2) return;
    ctx.beginPath();
    swipePoints.forEach((pt, idx) => {
      const x = ((pt.x + 1) / 2) * canvas.width;
      const y = ((pt.y + 1) / 2) * canvas.height;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();
  };

  const pointerDown = (e) => {
    e.preventDefault();
    const { normX, normY } = canvasToNormalized(canvas, e);
    pointerStart = { x: normX, y: normY, time: performance.now() };
    swipePoints = [];
    swipeActive = false;
    canvas.setPointerCapture(e.pointerId);
  };

  const pointerMove = (e) => {
    if (!pointerStart) return;
    const now = performance.now();
    const { normX, normY } = canvasToNormalized(canvas, e);
    if (!swipeActive) {
      const dist = Math.hypot(normX - pointerStart.x, normY - pointerStart.y);
      if (dist >= MIN_DRAG_DISTANCE || now - pointerStart.time >= MIN_DRAG_TIME) {
        swipeActive = true;
        swipePoints.push({ x: pointerStart.x, y: pointerStart.y, t: 0 });
      } else {
        return;
      }
    }
    swipePoints.push({ x: normX, y: normY, t: now - pointerStart.time });
    drawPath();
  };

  const pointerUp = async (e) => {
    canvas.releasePointerCapture(e.pointerId);
    if (!pointerStart) return;
    if (!swipeActive) {
      swipePoints = [
        { x: pointerStart.x, y: pointerStart.y, t: 0 },
        { x: pointerStart.x + 0.0005, y: pointerStart.y + 0.0005, t: 60 },
      ];
    }
    swipeActive = false;

    if (swipePoints.length > 1) {
      updateStatus('Processing swipe...');
      try {
        const result = await runtime.inferSwipe(swipePoints);
        showPrediction(result);
      } catch (err) {
        console.error(err);
        updateStatus('Swipe failed – see console');
      }
    }
    swipePoints = [];
    pointerStart = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  canvas.addEventListener('pointerdown', pointerDown);
  canvas.addEventListener('pointermove', pointerMove);
  canvas.addEventListener('pointerup', pointerUp);
  canvas.addEventListener('pointerleave', pointerUp);
}

function updateStatus(text) {
  const el = document.getElementById('status');
  if (el) el.textContent = text;
}

function showPrediction({ word, suggestions, metrics }) {
  const out = document.getElementById('prediction');
  const alt = document.getElementById('alternatives');
  out.textContent = word || '(no output)';
  alt.innerHTML = '';
  suggestions.filter((w) => w && w !== word).slice(0, 3).forEach((w) => {
    const btn = document.createElement('button');
    btn.textContent = w;
    btn.addEventListener('click', () => {
      out.textContent = w;
    });
    alt.appendChild(btn);
  });
  recordDebug(word, suggestions, metrics);
  updateStatus('Ready');
}

function recordDebug(word, suggestions, metrics) {
  const entry = {
    word,
    suggestions: suggestions.slice(0, 3),
    ...metrics,
    timestamp: new Date().toISOString(),
  };
  DEBUG.current = entry;
  DEBUG.history.unshift(entry);
  if (DEBUG.history.length > DEBUG_HISTORY_LIMIT) DEBUG.history.pop();
  renderDebug();
}

function renderDebug() {
  const panel = document.getElementById('debugPanel');
  if (!panel || !DEBUG.enabled || !DEBUG.current) return;

  const cur = DEBUG.current;
  const formatMs = (ms) => (ms ? ms.toFixed(2) : '0.00');
  document.getElementById('debugWord').textContent = cur.word || '–';
  document.getElementById('debugEncode').textContent = formatMs(cur.encodeMs);
  document.getElementById('debugDecode').textContent = formatMs(cur.decodeMs);
  document.getElementById('debugTotal').textContent = formatMs(cur.totalMs);
  document.getElementById('debugPoints').textContent = `${cur.rawPoints} / ${cur.resampledPoints}`;
  document.getElementById('debugFrames').textContent = cur.frames;
  document.getElementById('debugSteps').textContent = cur.stepCalls;
  document.getElementById('debugSuggestions').textContent = cur.suggestions.join(', ') || '–';

  const historyEl = document.getElementById('debugHistory');
  historyEl.innerHTML = '';
  DEBUG.history.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'debug-history-entry';
    row.innerHTML = `<strong>${idx + 1}. ${item.word || '∅'}</strong><span>${item.resampledPoints}p · ${item.frames}f · ${item.tokens || 0}tok · ${formatMs(item.totalMs)}ms</span>`;
    historyEl.appendChild(row);
  });
}

function toggleDebugPanel() {
  DEBUG.enabled = !DEBUG.enabled;
  const panel = document.getElementById('debugPanel');
  if (panel) {
    panel.classList.toggle('active', DEBUG.enabled);
  }
  const btn = document.getElementById('debugToggle');
  if (btn) {
    btn.textContent = DEBUG.enabled ? 'Hide Debug' : 'Toggle Debug';
  }
  renderDebug();
}

async function bootstrap() {
  updateStatus('Loading models & vocab...');
  await runtime.load();
  updateStatus('Ready');
  setupCanvas();
  document.getElementById('clearBtn').addEventListener('click', () => {
    swipePoints = [];
    pointerStart = null;
    const canvas = document.getElementById('swipeCanvas');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = '';
    document.getElementById('alternatives').innerHTML = '';
  });
  document.getElementById('debugToggle').addEventListener('click', toggleDebugPanel);
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('swipeCanvas');
  const resize = () => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  };
  window.addEventListener('resize', resize);
  resize();
  bootstrap().catch((err) => {
    console.error(err);
    updateStatus('Initialization failed');
  });
});
