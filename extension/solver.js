'use strict';

// Runs in the popup page after tf-core/layers/backend-cpu and preprocessing.js.
//
// Model output: [1, 50, 37] — 50 timesteps, 36 char classes + 1 CTC blank.
// CTC blank index: 36  (must match Python training: len(char_classes)==36, blank=36).
//
// Greedy CTC decode (TF.js has no built-in ctc_decode):
//   per-timestep argmax -> collapse consecutive duplicates -> drop blank (36).

const _BLANK = 36;

let _model   = null;
let _classes = null;

async function _loadModel() {
    if (_model) return;

    const modelUrl   = chrome.runtime.getURL('tfjs_model/model.json');
    const classesUrl = chrome.runtime.getURL('tfjs_model/char_classes.json');

    _model = await tf.loadLayersModel(modelUrl);

    const res = await fetch(classesUrl);
    _classes  = await res.json();

    console.log(`[BreakCAPTCHA] Model loaded. ${_classes.length} classes, blank=${_BLANK}`);
}

function _greedyCtcDecode(argmaxes) {
    const chars = [];
    let prev = -1;
    for (const c of argmaxes) {
        if (c !== prev && c !== _BLANK) chars.push(_classes[c]);
        prev = c;
    }
    return chars.join('');
}

/**
 * Run inference on pre-extracted RGBA pixel data from the content script.
 * @param {number[]} rgbaPixels  Flat RGBA array (200 * 32 * 4 values).
 * @returns {Promise<string>}    Decoded CAPTCHA text.
 */
async function solveFromPixels(rgbaPixels) {
    await _loadModel();

    const input        = preprocessFromPixels(rgbaPixels, 200, 32);  // [1, 32, 200, 1]
    const raw          = _model.predict(input);
    const output       = Array.isArray(raw) ? raw[0] : raw;          // [1, 50, 37]
    const softmax      = tf.squeeze(output, [0]);                     // [50, 37]
    const argmaxTensor = tf.argMax(softmax, 1);                       // [50]
    const argmaxes     = await argmaxTensor.data();

    tf.dispose([input, output, softmax, argmaxTensor]);
    if (Array.isArray(raw)) tf.dispose(raw);

    return _greedyCtcDecode(argmaxes);
}
