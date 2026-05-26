'use strict';

// Runs in the popup page after tf-core.min.js.
// Receives raw RGBA pixel data (already resized to 200x32 by content.js canvas)
// and returns a [1, 32, 200, 1] float32 tensor.
//
// Grayscale formula matches Python preprocess.py exactly:
//   cv2.COLOR_BGR2GRAY = 0.114*B + 0.587*G + 0.299*R
//   Canvas pixel order: R, G, B, A  →  0.299*R + 0.587*G + 0.114*B

/**
 * @param {number[]} rgbaPixels  Flat RGBA array, length = width * height * 4.
 * @param {number}   width       Should be 200.
 * @param {number}   height      Should be 32.
 * @returns {tf.Tensor4D}  Shape [1, height, width, 1], float32. Caller must dispose.
 */
function preprocessFromPixels(rgbaPixels, width, height) {
    const gray = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
        gray[i] = (
            0.299 * rgbaPixels[i * 4]     +   // R
            0.587 * rgbaPixels[i * 4 + 1] +   // G
            0.114 * rgbaPixels[i * 4 + 2]     // B
        ) / 255;
    }
    return tf.tensor(gray, [1, height, width, 1], 'float32');
}
