/**
 * Feature Extractor Module - Corrected to match training
 * Handles coordinate normalization and feature extraction for swipe gestures
 */

class SwipeFeatureExtractorCorrected {
    constructor() {
        this.featureDim = 37;
        // Build key centers in [0, 1] coordinates matching training
        this.keyCenters = this.buildKeyCenters();
    }

    /**
     * Build keyboard key centers in [-1, 1] coordinates
     */
    buildKeyCenters() {
        const layout = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ];

        const centers = [];
        for (let row = 0; row < layout.length; row++) {
            const rowStr = layout[row];
            for (let col = 0; col < rowStr.length; col++) {
                const char = rowStr[col];
                // Calculate position in [0, 1]
                const x = (col + 0.5) / 10.0;  // 10 keys max width
                const y = (row + 0.5) / 3.0;   // 3 rows
                centers.push({ char, x, y });
            }
        }
        return centers;
    }

    /**
     * Normalize points to [0, 1] coordinate system used in training.
     * If points appear in [-1, 1], map to [0, 1]. If already [0, 1], clamp.
     */
    normalizePoints(points) {
        if (!points || points.length === 0) {
            return [];
        }
        const startTime = points[0].t || 0;
        
        // Heuristic: if coordinates appear in [-1,1], convert to [0,1]; else assume already [0,1]
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of points) {
            if (typeof p.x === 'number') { if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x; }
            if (typeof p.y === 'number') { if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y; }
        }
        const likelyMinus1to1 = (minX >= -1 && maxX <= 1 && (minX < 0 || maxX > 1));

        return points.map((pt, idx) => {
            let x = pt.x ?? 0.0;
            let y = pt.y ?? 0.0;
            if (likelyMinus1to1) {
                x = (x + 1.0) * 0.5;
                y = (y + 1.0) * 0.5;
            }
            const centeredX = Math.max(0.0, Math.min(1.0, x));
            const centeredY = Math.max(0.0, Math.min(1.0, y));
            const t = (pt.t || idx * 10.0) - startTime;
            return { x: centeredX, y: centeredY, t };
        });
    }

    /**
     * Determine adaptive resample target based on trace length
     */
    getResampleTarget(length) {
        const shortTarget = 56, longTarget = 96, shortThresh = 48, longThresh = 112;
        if (length <= shortThresh) return shortTarget;
        if (length >= longThresh) return longTarget;
        const progress = (length - shortThresh) / (longThresh - shortThresh);
        return Math.round(shortTarget + progress * (longTarget - shortTarget));
    }

    /**
     * Resample points to target count using linear interpolation
     */
    resamplePoints(points, targetCount) {
        if (targetCount <= 0 || points.length === 0) return [];
        if (points.length === targetCount) return [...points];
        const resampled = [];
        const duration = points.length > 1 ? points[points.length - 1].t - points[0].t : 0;
        const step = duration / Math.max(targetCount - 1, 1);
        let srcIdx = 0;
        for (let i = 0; i < targetCount; i++) {
            const targetTime = (points[0].t || 0) + step * i;
            while (srcIdx < points.length - 2 && points[srcIdx + 1].t < targetTime) {
                srcIdx++;
            }
            const p1 = points[srcIdx];
            const p2 = points[Math.min(srcIdx + 1, points.length - 1)];
            const span = Math.max(p2.t - p1.t, 1e-6);
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
     * Extract features from a single point in context
     */
    extractPointFeatures(points, idx) {
        const total = points.length;
        const curr = points[idx];
        const prev = idx > 0 ? points[idx - 1] : null;
        const prev2 = idx > 1 ? points[idx - 2] : null;

        const x = curr.x, y = curr.y, t_seconds = curr.t / 1000.0;
        let vx = 0, vy = 0, speed = 0;
        if (prev) {
            const dt = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            vx = (x - prev.x) / dt;
            vy = (y - prev.y) / dt;
            speed = Math.sqrt(vx * vx + vy * vy);
        }

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

        const angle = prev ? Math.atan2(vy, vx) : 0.0;
        let curvature = 0;
        if (prev && prev2) {
            const prev_angle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
            curvature = angle - prev_angle;
            while (curvature > Math.PI) curvature -= 2 * Math.PI;
            while (curvature < -Math.PI) curvature += 2 * Math.PI;
        }

        const keyDistances = this.keyCenters
            .map(key => Math.sqrt((x - key.x) ** 2 + (y - key.y) ** 2))
            .sort((a, b) => a - b)
            .slice(0, 5);
        while (keyDistances.length < 5) keyDistances.push(1.0);

        const progress = idx / Math.max(total - 1, 1);
        const is_start = idx === 0 ? 1.0 : 0.0;
        const is_end = idx === total - 1 ? 1.0 : 0.0;

        const winStart = Math.max(0, idx - 2);
        const winEnd = Math.min(total, idx + 3);
        const winPts = points.slice(winStart, winEnd);
        let win_mean_x = x, win_std_x = 0, win_mean_y = y, win_std_y = 0, win_range_x = 0, win_range_y = 0;
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

        const features = [
            x, y, t_seconds, vx, vy, speed, ax, ay, acc, angle, Math.sin(angle), Math.cos(angle), curvature,
            ...keyDistances, progress, is_start, is_end,
            win_mean_x, win_std_x, win_mean_y, win_std_y, win_range_x, win_range_y
        ];
        while (features.length < this.featureDim) features.push(0.0);
        return features.slice(0, this.featureDim);
    }

    /**
     * Process swipe trace into features matching training pipeline
     * @param {Array} rawPoints - Raw swipe points in [-1, 1] coordinates
     * @returns {Object} Features and metadata
     */
    process(rawPoints) {
        const normalizedPoints = this.normalizePoints(rawPoints);
        const targetLength = this.getResampleTarget(normalizedPoints.length);
        const resampledPoints = this.resamplePoints(normalizedPoints, targetLength);
        const featureMatrix = resampledPoints.map((_, idx) =>
            this.extractPointFeatures(resampledPoints, idx)
        );

        const numFrames = featureMatrix.length;
        const flatFeatures = new Float32Array(numFrames * this.featureDim);
        for (let t = 0; t < numFrames; t++) {
            flatFeatures.set(featureMatrix[t], t * this.featureDim);
        }

        return {
            features: flatFeatures,
            featureMatrix: featureMatrix,
            numFrames: numFrames,
            originalLength: rawPoints.length,
            duration: normalizedPoints.length > 0 ?
                normalizedPoints[normalizedPoints.length - 1].t - normalizedPoints[0].t : 0
        };
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SwipeFeatureExtractorCorrected;
}
