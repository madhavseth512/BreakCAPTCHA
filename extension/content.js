'use strict';

// Lightweight content script — no TF.js here.
// Inference runs in the popup page (normal HTML context) where TF.js loads reliably.
// This script does two things only:
//   1. Extract CAPTCHA image pixels (getImageData) and send them to the popup.
//   2. Fill the CAPTCHA input with the solved text (fillInput).

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.action === 'getImageData') {
        sendResponse(_extractImageData());
        return false;
    }
    if (msg.action === 'fillInput') {
        _fillInput(msg.text);
        sendResponse({ ok: true });
        return false;
    }
});

function _extractImageData() {
    const img = _findCaptchaImage();
    if (!img)                                  return { ok: false, error: 'No CAPTCHA image found on this page.' };
    if (!img.complete || !img.naturalWidth)    return { ok: false, error: 'CAPTCHA image has not finished loading.' };

    const canvas = document.createElement('canvas');
    canvas.width  = 200;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 200, 32);

    let imageData;
    try {
        imageData = ctx.getImageData(0, 0, 200, 32);
    } catch (e) {
        return { ok: false, error: 'Cannot read CAPTCHA pixels (cross-origin image?).' };
    }

    // Convert Uint8ClampedArray -> plain Array for structured-clone transfer.
    return { ok: true, pixels: Array.from(imageData.data), width: 200, height: 32 };
}

function _fillInput(text) {
    const input = _findCaptchaInput();
    if (!input) return;
    input.value = text;
    input.dispatchEvent(new Event('input',  { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
}

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

function _findCaptchaInput() {
    return (
        document.querySelector('input.captcha-input')   ||
        document.querySelector('input[name="captcha"]') ||
        document.querySelector('input[type="text"]')    ||
        document.querySelector('input:not([type])')     ||
        null
    );
}
