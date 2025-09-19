// Performance Metrics Tracking
class PerformanceMetrics {
    constructor() {
        this.inferenceHistory = [];
        this.gestureCount = 0;
        this.startTime = Date.now();
        this.wordCount = 0;
        this.lastWordTime = Date.now();
    }

    recordInference(latencyMs) {
        this.inferenceHistory.push(latencyMs);
        // Keep only last 10 measurements
        if (this.inferenceHistory.length > 10) {
            this.inferenceHistory.shift();
        }
        this.updateDisplay();
    }

    recordGesture() {
        this.gestureCount++;
        this.updateDisplay();
    }

    recordWord() {
        this.wordCount++;
        this.lastWordTime = Date.now();
        this.updateDisplay();
    }

    getAverageInference() {
        if (this.inferenceHistory.length === 0) return 0;
        const sum = this.inferenceHistory.reduce((a, b) => a + b, 0);
        return sum / this.inferenceHistory.length;
    }

    getWordsPerMinute() {
        const timeElapsed = (Date.now() - this.startTime) / 1000 / 60; // minutes
        if (timeElapsed < 0.1) return 0; // Need at least 6 seconds
        return Math.round(this.wordCount / timeElapsed);
    }

    updateDisplay() {
        // Update metrics display
        const metricsDiv = document.getElementById('metrics');
        if (metricsDiv) {
            metricsDiv.classList.remove('hidden');

            const inferenceEl = document.getElementById('inferenceTime');
            const wpmEl = document.getElementById('wordsPerMinute');
            const gestureEl = document.getElementById('gestureCount');

            if (inferenceEl) {
                inferenceEl.textContent = this.getAverageInference().toFixed(1);
            }
            if (wpmEl) {
                wpmEl.textContent = this.getWordsPerMinute();
            }
            if (gestureEl) {
                gestureEl.textContent = this.gestureCount;
            }
        }
    }

    reset() {
        this.inferenceHistory = [];
        this.gestureCount = 0;
        this.wordCount = 0;
        this.startTime = Date.now();
        this.lastWordTime = Date.now();
        this.updateDisplay();
    }
}

// Model Preloading with IndexedDB Cache
class ModelPreloader {
    constructor() {
        this.dbName = 'swipe-model-cache';
        this.dbVersion = 1;
        this.storeName = 'models';
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName);
                }
            };
        });
    }

    async cacheModel(modelPath, arrayBuffer) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.put(arrayBuffer, modelPath);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async getCachedModel(modelPath) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(modelPath);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async preloadModels(modelPaths, onProgress) {
        const results = [];

        for (let i = 0; i < modelPaths.length; i++) {
            const modelPath = modelPaths[i];

            try {
                // Check if already cached
                const cached = await this.getCachedModel(modelPath);
                if (cached) {
                    console.log(`✓ Model ${modelPath} already cached`);
                    results.push({ path: modelPath, cached: true });
                    if (onProgress) onProgress(i + 1, modelPaths.length, modelPath, true);
                    continue;
                }

                // Fetch and cache
                console.log(`⬇ Downloading ${modelPath}...`);
                const response = await fetch(modelPath);
                const arrayBuffer = await response.arrayBuffer();

                await this.cacheModel(modelPath, arrayBuffer);
                console.log(`✓ Cached ${modelPath} (${(arrayBuffer.byteLength / 1024 / 1024).toFixed(1)}MB)`);

                results.push({ path: modelPath, cached: false, size: arrayBuffer.byteLength });
                if (onProgress) onProgress(i + 1, modelPaths.length, modelPath, false);

            } catch (error) {
                console.error(`✗ Failed to preload ${modelPath}:`, error);
                results.push({ path: modelPath, error: error.message });
            }
        }

        return results;
    }

    async loadModelFromCache(modelPath, sessionOptions) {
        try {
            const arrayBuffer = await this.getCachedModel(modelPath);
            if (arrayBuffer) {
                console.log(`📦 Loading ${modelPath} from cache`);
                return await ort.InferenceSession.create(arrayBuffer, sessionOptions);
            }
        } catch (error) {
            console.warn(`Cache miss for ${modelPath}, loading from network`);
        }

        // Fallback to network
        return await ort.InferenceSession.create(modelPath, sessionOptions);
    }
}

// Gesture Recording for Dataset Collection
class GestureRecorder {
    constructor() {
        this.recordings = [];
        this.currentRecording = null;
        this.isRecording = false;
    }

    startRecording(word) {
        this.currentRecording = {
            word: word,
            timestamp: Date.now(),
            points: [],
            features: null,
            keySequence: [],
            modelUsed: null,
            inferenceTime: null
        };
        this.isRecording = true;
    }

    addPoint(point) {
        if (this.isRecording && this.currentRecording) {
            this.currentRecording.points.push({
                x: point.x,
                y: point.y,
                t: point.t || Date.now() - this.currentRecording.timestamp
            });
        }
    }

    addKeyPress(key) {
        if (this.isRecording && this.currentRecording) {
            this.currentRecording.keySequence.push(key);
        }
    }

    finishRecording(features, modelPath, inferenceTime) {
        if (this.isRecording && this.currentRecording) {
            this.currentRecording.features = features;
            this.currentRecording.modelUsed = modelPath;
            this.currentRecording.inferenceTime = inferenceTime;
            this.currentRecording.duration = Date.now() - this.currentRecording.timestamp;

            this.recordings.push(this.currentRecording);
            this.currentRecording = null;
            this.isRecording = false;

            // Save to localStorage for persistence
            this.saveToStorage();

            return this.recordings[this.recordings.length - 1];
        }
        return null;
    }

    cancelRecording() {
        this.currentRecording = null;
        this.isRecording = false;
    }

    saveToStorage() {
        try {
            // Keep only last 100 recordings
            if (this.recordings.length > 100) {
                this.recordings = this.recordings.slice(-100);
            }
            localStorage.setItem('gesture_recordings', JSON.stringify(this.recordings));
        } catch (error) {
            console.warn('Failed to save recordings to localStorage:', error);
        }
    }

    loadFromStorage() {
        try {
            const saved = localStorage.getItem('gesture_recordings');
            if (saved) {
                this.recordings = JSON.parse(saved);
                console.log(`Loaded ${this.recordings.length} recordings from storage`);
            }
        } catch (error) {
            console.warn('Failed to load recordings from localStorage:', error);
        }
    }

    exportRecordings() {
        const data = {
            version: '1.0',
            exportDate: new Date().toISOString(),
            recordings: this.recordings
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `gesture_recordings_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`Exported ${this.recordings.length} recordings`);
    }

    getStatistics() {
        const stats = {
            totalRecordings: this.recordings.length,
            uniqueWords: new Set(this.recordings.map(r => r.word)).size,
            averagePoints: 0,
            averageDuration: 0,
            averageInference: 0
        };

        if (this.recordings.length > 0) {
            const totals = this.recordings.reduce((acc, r) => {
                acc.points += r.points.length;
                acc.duration += r.duration || 0;
                acc.inference += r.inferenceTime || 0;
                return acc;
            }, { points: 0, duration: 0, inference: 0 });

            stats.averagePoints = Math.round(totals.points / this.recordings.length);
            stats.averageDuration = Math.round(totals.duration / this.recordings.length);
            stats.averageInference = (totals.inference / this.recordings.length).toFixed(1);
        }

        return stats;
    }
}

// Export for use in main HTML
window.PerformanceMetrics = PerformanceMetrics;
window.ModelPreloader = ModelPreloader;
window.GestureRecorder = GestureRecorder;

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.perfMetrics = new PerformanceMetrics();
    window.modelPreloader = new ModelPreloader();
    window.gestureRecorder = new GestureRecorder();

    // Load previous recordings if any
    window.gestureRecorder.loadFromStorage();

    // Start preloading models in background
    const models = [
        'encoder_web_ultra.onnx',
        'encoder_android_int8_final.onnx',
        'encoder_android_ultra.onnx',
        'encoder_fp32.onnx'
    ];

    console.log('🚀 Starting background model preloading...');
    window.modelPreloader.preloadModels(models, (current, total, path, cached) => {
        const status = cached ? 'cached' : 'downloaded';
        console.log(`[${current}/${total}] ${path} - ${status}`);
    }).then(results => {
        console.log('✅ Model preloading complete:', results);
    });
});