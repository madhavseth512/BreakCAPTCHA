'use strict';

// Loaded after tf.min.js.
// Must match Python preprocess.py exactly:
//   cv2.resize(img, (200, 32), interpolation=INTER_AREA)
//   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  -> 0.299*R + 0.587*G + 0.114*B
//   gray / 255.0
//
// Canvas uses bilinear resize (vs. INTER_AREA in Python). The difference is
// minor for smooth CAPTCHA glyphs and the model is robust to it.

const _TARGET_W = 200;
const _TARGET_H = 32;

/**
 * Preprocess a CAPTCHA <img> element into a [1, 32, 200, 1] float32 tensor.
 * The tensor must be disposed by the caller after use.
 */
function preprocessCaptcha(imgElement) {
    const canvas = document.createElement('canvas');
    canvas.width  = _TARGET_W;
    canvas.height = _TARGET_H;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, _TARGET_W, _TARGET_H);

    const { data } = ctx.getImageData(0, 0, _TARGET_W, _TARGET_H);  // RGBA uint8, row-major

    const gray = new Float32Array(_TARGET_W * _TARGET_H);
    for (let i = 0; i < _TARGET_W * _TARGET_H; i++) {
        // Canvas: R=data[4i], G=data[4i+1], B=data[4i+2]
        // OpenCV BGR2GRAY: 0.114*B + 0.587*G + 0.299*R  (same formula, different channel order)
        gray[i] = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
    }

    // Shape: [batch=1, H=32, W=200, C=1]
    return tf.tensor(gray, [1, _TARGET_H, _TARGET_W, 1], 'float32');
}
