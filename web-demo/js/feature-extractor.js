/**
 * Feature Extractor Module
 * Handles coordinate normalization and feature extraction for swipe gestures
 */

class SwipeFeatureExtractor {
    constructor(keyboardLayout = null) {
        this.keyCenters = keyboardLayout || this.getDefaultQWERTYLayout();
        this.featureDim = 37;
    }

    /**
     * Get default QWERTY keyboard layout with proper coordinates
     */
    getDefaultQWERTYLayout() {
        const layout = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ];

        const centers = [];
        const keyWidth = 1.0 / 10.0;  // 10 keys in top row
        const keyHeight = 1.0 / 3.0;  // 3 rows

        for (let row = 0; row < layout.length; row++) {
            const rowStr = layout[row];
            const rowOffset = row === 2 ? 0.5 * keyWidth : 0; // Offset for bottom row

            for (let col = 0; col < rowStr.length; col++) {
                const char = rowStr[col];
                // Convert from [0,1] to [-1,1] coordinate system
                const x01 = rowOffset + (col + 0.5) * keyWidth;
                const y01 = (row + 0.5) * keyHeight;
                const x = x01 * 2.0 - 1.0;
                const y = y01 * 2.0 - 1.0;
                centers.push({ char, x, y });
            }
        }
        return centers;
    }

    /**
     * Normalize canvas coordinates to [-1, 1] range
     * Canvas origin is top-left, model expects center origin
     */
    normalizeCoordinates(canvasX, canvasY, canvasWidth, canvasHeight) {
        // Convert from canvas coordinates to [-1, 1]
        // Canvas: (0, 0) at top-left, (width, height) at bottom-right
        // Model: (-1, -1) at top-left, (1, 1) at bottom-right
        const x = (canvasX / canvasWidth) * 2.0 - 1.0;
        const y = (canvasY / canvasHeight) * 2.0 - 1.0;
        return { x, y };
    }

    /**
     * Prepare points from raw swipe data
     * @param {Array} points - Array of {x, y, t} points
     * @returns {Array} Normalized points
     */
    preparePoints(points) {
        if (!points || points.length === 0) {
            return [];
        }

        const startTime = points[0].t || 0;
        return points.map((pt, idx) => ({
            x: Math.max(-1.0, Math.min(1.0, pt.x)),
            y: Math.max(-1.0, Math.min(1.0, pt.y)),
            t: (pt.t || idx * 10.0) - startTime
        }));
    }

    /**
     * Resample points to target count using linear interpolation
     */
    resamplePoints(points, targetCount) {
        if (targetCount <= 0 || points.length === 0) return [];
        if (points.length === targetCount) return [...points];

        const resampled = [];
        const duration = points[points.length - 1].t - points[0].t;
        const step = duration / Math.max(targetCount - 1, 1);

        let srcIdx = 0;
        for (let i = 0; i < targetCount; i++) {
            const targetTime = points[0].t + step * i;

            // Find surrounding points
            while (srcIdx < points.length - 2 && points[srcIdx + 1].t < targetTime) {
                srcIdx++;
            }

            const p1 = points[srcIdx];
            const p2 = points[Math.min(srcIdx + 1, points.length - 1)];
            const span = Math.max(p2.t - p1.t, 1.0);
            const alpha = Math.max(0, Math.min(1, (targetTime - p1.t) / span));

            resampled.push({
                x: p1.x + (p2.x - p1.x) * alpha,
                y: p1.y + (p2.y - p1.y) * alpha,
                t: targetTime
            });
        }

        return resampled;
    }

    /**
     * Determine adaptive resample target based on trace length
     */
    getResampleTarget(length) {
        const shortTarget = 56;
        const longTarget = 96;
        const shortThresh = 48;
        const longThresh = 112;

        if (length <= shortThresh) return shortTarget;
        if (length >= longThresh) return longTarget;

        // Linear interpolation
        const progress = (length - shortThresh) / (longThresh - shortThresh);
        return Math.round(shortTarget + progress * (longTarget - shortTarget));
    }

    /**
     * Extract features from a single point in context
     */
    extractPointFeatures(points, idx) {
        const total = points.length;
        const curr = points[idx];
        const prev = idx > 0 ? points[idx - 1] : null;
        const prev2 = idx > 1 ? points[idx - 2] : null;

        // Basic position and time
        const x = curr.x;
        const y = curr.y;
        const t_seconds = curr.t / 1000.0;

        // Velocity
        let vx = 0, vy = 0, speed = 0;
        if (prev) {
            const dt = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            vx = (x - prev.x) / dt;
            vy = (y - prev.y) / dt;
            speed = Math.sqrt(vx * vx + vy * vy);
        }

        // Acceleration
        let ax = 0, ay = 0, acc = 0;
        if (prev && prev2) {
            const dt1 = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            const dt2 = Math.max((prev.t - prev2.t) / 1000.0, 1e-6);
            const vx_prev = (prev.x - prev2.x) / dt2;
            const vy_prev = (prev.y - prev2.y) / dt2;
            ax = (vx - vx_prev) / dt1;
            ay = (vy - vy_prev) / dt1;
            acc = Math.sqrt(ax * ax + ay * ay);
        }

        // Angle and curvature
        const angle = prev ? Math.atan2(vy, vx) : 0.0;
        let curvature = 0;
        if (prev && prev2) {
            const prev_angle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
            curvature = angle - prev_angle;
            while (curvature > Math.PI) curvature -= 2 * Math.PI;
            while (curvature < -Math.PI) curvature += 2 * Math.PI;
        }

        // Distance to nearest keys
        const keyDistances = this.keyCenters
            .map(key => Math.sqrt((x - key.x) ** 2 + (y - key.y) ** 2))
            .sort((a, b) => a - b)
            .slice(0, 5);
        while (keyDistances.length < 5) keyDistances.push(1.0);

        // Progress and position markers
        const progress = idx / Math.max(total - 1, 1);
        const is_start = idx === 0 ? 1.0 : 0.0;
        const is_end = idx === total - 1 ? 1.0 : 0.0;

        // Window statistics
        const winStart = Math.max(0, idx - 2);
        const winEnd = Math.min(total, idx + 3);
        const winPts = points.slice(winStart, winEnd);

        let win_mean_x = x, win_std_x = 0, win_mean_y = y, win_std_y = 0;
        let win_range_x = 0, win_range_y = 0;

        if (winPts.length > 1) {
            const xs = winPts.map(p => p.x);
            const ys = winPts.map(p => p.y);
            win_mean_x = xs.reduce((a, b) => a + b, 0) / xs.length;
            win_mean_y = ys.reduce((a, b) => a + b, 0) / ys.length;
            win_std_x = Math.sqrt(xs.reduce((sum, xi) => sum + (xi - win_mean_x) ** 2, 0) / xs.length);
            win_std_y = Math.sqrt(ys.reduce((sum, yi) => sum + (yi - win_mean_y) ** 2, 0) / ys.length);
            win_range_x = Math.max(...xs) - Math.min(...xs);
            win_range_y = Math.max(...ys) - Math.min(...ys);
        }

        // Assemble 37-dimensional feature vector
        const features = [
            x, y, t_seconds,                                    // 3: position and time
            vx, vy, speed,                                       // 6: velocity
            ax, ay, acc,                                         // 9: acceleration
            angle, Math.sin(angle), Math.cos(angle), curvature, // 13: trajectory shape
            ...keyDistances,                                    // 18: spatial context (5 nearest keys)
            progress, is_start, is_end,                         // 21: temporal context
            win_mean_x, win_std_x, win_mean_y, win_std_y,      // 25: window statistics
            win_range_x, win_range_y                           // 27: window range
        ];

        // Pad to 37 features
        while (features.length < this.featureDim) {
            features.push(0.0);
        }

        return features.slice(0, this.featureDim);
    }

    /**
     * Extract features from a complete swipe gesture
     * @param {Array} rawPoints - Raw swipe points from canvas
     * @param {Object} options - Processing options
     * @returns {Object} Feature matrix and metadata
     */
    extractFeatures(rawPoints, options = {}) {
        // Prepare and normalize points
        const normalizedPoints = this.preparePoints(rawPoints);

        // Determine resample target
        const targetLength = options.targetLength || this.getResampleTarget(normalizedPoints.length);

        // Resample points
        const resampledPoints = this.resamplePoints(normalizedPoints, targetLength);

        // Extract features for each point
        const featureMatrix = resampledPoints.map((_, idx) =>
            this.extractPointFeatures(resampledPoints, idx)
        );

        return {
            features: featureMatrix,
            originalLength: rawPoints.length,
            resampledLength: resampledPoints.length,
            duration: normalizedPoints.length > 0 ?
                normalizedPoints[normalizedPoints.length - 1].t - normalizedPoints[0].t : 0
        };
    }
}

/**
 * Preprocessor wrapper for compatibility
 */
class SwipePreprocessor {
    constructor() {
        this.extractor = new SwipeFeatureExtractor();
    }

    /**
     * Process swipe trace into features
     * @param {Array} points - Array of {x, y, t} points
     * @returns {Object} Features and metadata
     */
    process(points) {
        const result = this.extractor.extractFeatures(points);
        // Flatten feature matrix for ONNX input
        const numFrames = result.features.length;
        const featureDim = 37;
        const flatFeatures = new Float32Array(numFrames * featureDim);

        for (let t = 0; t < numFrames; t++) {
            for (let f = 0; f < featureDim; f++) {
                flatFeatures[t * featureDim + f] = result.features[t][f];
            }
        }

        return {
            features: flatFeatures,
            numFrames: numFrames,
            originalLength: result.originalLength,
            duration: result.duration
        };
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SwipeFeatureExtractor, SwipePreprocessor };
}