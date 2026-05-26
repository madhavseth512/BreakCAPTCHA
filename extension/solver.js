'use strict';

// Loaded after tf.min.js and preprocessing.js.
//
// Model output: [1, 50, 37]  — 50 timesteps, 36 char classes + 1 CTC blank.
// CTC blank index: 36  (must match Python: len(char_classes) == 36, blank = 36).
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

/**
 * Greedy CTC decode on a flat number[] of per-timestep argmax values.
 * Returns the decoded string (up to 4 chars for this model).
 */
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
 * Solve a CAPTCHA <img> element. Returns a Promise<string> with the decoded text.
 * Tensors are disposed internally; no cleanup needed by caller.
 */
async function solveImage(imgElement) {
    await _loadModel();

    const input     = preprocessCaptcha(imgElement);          // [1, 32, 200, 1]
    const output    = _model.predict(input);                   // [1, 50, 37]
    const softmax   = output.squeeze([0]);                     // [50, 37]
    const argmaxTensor = tf.argMax(softmax, 1);                // [50] — argmax over classes
    const argmaxes  = await argmaxTensor.array();              // number[50]

    tf.dispose([input, output, softmax, argmaxTensor]);

    return _greedyCtcDecode(argmaxes);
}
