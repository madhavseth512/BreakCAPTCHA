'use strict';

// Loaded after tf.min.js, preprocessing.js, solver.js.
// Listens for {action: "solve"} from popup.js and solves the CAPTCHA on the page.

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.action !== 'solve') return;

    _solveCurrentPage()
        .then(result => sendResponse({ ok: true,  result }))
        .catch(err   => sendResponse({ ok: false, error: err.message }));

    return true;  // keep the message channel open for the async response
});

async function _solveCurrentPage() {
    const img = _findCaptchaImage();
    if (!img) throw new Error('No CAPTCHA image found on this page.');

    // Wait for the image to finish loading (needed if src was just set).
    await _waitForImage(img);

    const text = await solveImage(img);

    const input = _findCaptchaInput();
    if (input) {
        input.value = text;
        input.dispatchEvent(new Event('input',  { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }

    return text;
}

/**
 * Find the CAPTCHA image on the page.
 * Priority order: explicit class/id → any img whose id/class/src contains "captcha".
 */
function _findCaptchaImage() {
    return (
        document.querySelector('img.captcha-image') ||
        document.querySelector('img#captcha')       ||
        [...document.querySelectorAll('img')].find(el => {
            const hay = `${el.id} ${el.className} ${el.src}`.toLowerCase();
            return hay.includes('captcha');
        }) ||
        null
    );
}

/**
 * Find the text input to fill in.
 * Priority: explicit captcha class/name → any visible text input.
 */
function _findCaptchaInput() {
    return (
        document.querySelector('input.captcha-input')   ||
        document.querySelector('input[name="captcha"]') ||
        document.querySelector('input[type="text"]')    ||
        document.querySelector('input:not([type])')     ||
        null
    );
}

function _waitForImage(img) {
    if (img.complete && img.naturalWidth > 0) return Promise.resolve();
    return new Promise((resolve, reject) => {
        img.onload  = resolve;
        img.onerror = () => reject(new Error('CAPTCHA image failed to load.'));
    });
}
