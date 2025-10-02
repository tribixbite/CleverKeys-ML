/**
 * Feature Extractor Module - Corrected to match training
 * Handles coordinate normalization and feature extraction for swipe gestures
 */

class SwipeFeatureExtractorCorrected {
    constructor() {
        this.featureDim = 37;
        this.keyCenters = [];
        this.verbose = false;
    }

    /**
     * Normalize points to [0, 1] coordinate system used in training.
     * If incoming coordinates are in [-1,1], convert to [0,1].
     * This auto-detects ranges robustly (min < -0.05 or max > 1.05 -> treat as [-1,1]).
     */
    normalizePoints(points) {
        if (!points || points.length === 0) {
            return [];
        }
        const startTime = points[0].t || 0;

        const normed = points.map((pt, idx) => {
            return {
                x: pt.x ?? 0.0,
                y: pt.y ?? 0.0,
                t: (pt.t || idx * 10.0) - startTime
            };
        });

        if (this.verbose) {
            const xs = normed.map(p => p.x), ys = normed.map(p => p.y);
            const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
            console.log('[FE] normalizePoints: input range x=[%d,%d] y=[%d,%d] n=%d',
                +minX.toFixed?.(3) || minX, +maxX.toFixed?.(3) || maxX,
                +minY.toFixed?.(3) || minY, +maxY.toFixed?.(3) || maxY,
                normed.length);
        }
        return normed;
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
            speed = Math.hypot(vx, vy);
        }

        let ax = 0, ay = 0, acc = 0;
        if (prev && prev2) {
            const dt1 = Math.max((curr.t - prev.t) / 1000.0, 1e-6);
            const dt2 = Math.max((prev.t - prev2.t) / 1000.0, 1e-6);
            const vx_prev = (prev.x - prev2.x) / dt2;
            const vy_prev = (prev.y - prev2.y) / dt2;
            ax = (vx - vx_prev) / dt1;
            ay = (vy - vy_prev) / dt1;
            acc = Math.hypot(ax, ay);
        }

        const angle = prev ? Math.atan2(vy, vx) : 0.0;
        let curvature = 0;
        if (prev && prev2) {
            const prev_angle = Math.atan2(prev.y - prev2.y, prev.x - prev2.x);
            curvature = angle - prev_angle;
            while (curvature > Math.PI) curvature -= 2 * Math.PI;
            while (curvature < -Math.PI) curvature += 2 * Math.PI;
        }

        const keyDistances = this.keyCenters.map(key => Math.hypot(x - key.x, y - key.y))
            .sort((a, b) => a - b).slice(0, 5);
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

        // Match the Python feature dictionary and ordering
        const featureDict = {
            "x": x,
            "y": y,
            "t_seconds": t_seconds,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "ax": ax,
            "ay": ay,
            "acc": acc,
            "angle": angle,
            "angle_sin": Math.sin(angle),
            "angle_cos": Math.cos(angle),
            "curvature": curvature,
            "dist_key1": keyDistances[0],
            "dist_key2": keyDistances[1],
            "dist_key3": keyDistances[2],
            "dist_key4": keyDistances[3],
            "dist_key5": keyDistances[4],
            "progress": progress,
            "is_start": is_start,
            "is_end": is_end,
            "win_mean_x": win_mean_x,
            "win_std_x": win_std_x,
            "win_mean_y": win_mean_y,
            "win_std_y": win_std_y,
            "win_range_x": win_range_x,
            "win_range_y": win_range_y,
        };

        // This order MUST match PersonalizedSwipeFeaturizer.FULL_FEATURE_NAMES
        const featureNames = [
            "x", "y", "t_seconds", "vx", "vy", "speed", "ax", "ay", "acc", "angle",
            "angle_sin", "angle_cos", "curvature", "dist_key1", "dist_key2", "dist_key3",
            "dist_key4", "dist_key5", "progress", "is_start", "is_end", "win_mean_x",
            "win_std_x", "win_mean_y", "win_std_y", "win_range_x", "win_range_y"
        ];

        const featureVector = featureNames.map(name => featureDict[name] || 0.0);

        // Pad to final dimension, matching training
        while (featureVector.length < this.featureDim) {
            featureVector.push(0.0);
        }

        if (this.verbose) {
            const nearest = this.keyCenters.map(k => ({ char: k.char, dist: Math.hypot(x - k.x, y - k.y) })).sort((a,b)=>a.dist-b.dist)[0]?.char || '?';
            console.log('[FE] t#%d x=%.3f y=%.3f vx=%.3f vy=%.3f speed=%.3f near=%s', idx, x, y, vx, vy, speed, nearest);
        }
        return featureVector.slice(0, this.featureDim);
    }

    /**
     * Process swipe trace into features matching training pipeline
     * @param {Array} rawPoints - Raw swipe points in [-1, 1] coordinates
     * @returns {Object} Features and metadata
     */
    process(rawPoints) {
        const normalizedPoints = this.normalizePoints(rawPoints);
        // Adaptive resampling to match training (56–96 frames typical)
        const target = this.getResampleTarget(normalizedPoints.length);
        const pts = this.resamplePoints(normalizedPoints, target);
        const featureMatrix = pts.map((_, idx) => this.extractPointFeatures(pts, idx));

        const numFrames = featureMatrix.length;
        const flatFeatures = new Float32Array(numFrames * this.featureDim);
        for (let t = 0; t < numFrames; t++) {
            flatFeatures.set(featureMatrix[t], t * this.featureDim);
        }

        const result = {
            features: flatFeatures,
            featureMatrix: featureMatrix,
            numFrames: numFrames,
            originalLength: rawPoints.length,
            duration: normalizedPoints.length > 0 ?
                normalizedPoints[normalizedPoints.length - 1].t - normalizedPoints[0].t : 0
        };
        if (this.verbose) {
            console.log('[FE] frames=%d featDim=%d duration_ms=%d', numFrames, this.featureDim, result.duration|0);
            if (numFrames > 0) console.log('[FE] firstFrame=', featureMatrix[0].map(v=>+v.toFixed(4)));
        }
        return result;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SwipeFeatureExtractorCorrected;
}
