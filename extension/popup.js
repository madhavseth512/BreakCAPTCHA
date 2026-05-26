'use strict';

// TF.js, preprocessing.js, and solver.js are loaded as <script> tags in popup.html
// before this file, so tf / preprocessFromPixels / solveFromPixels are all available.

document.getElementById('solveBtn').addEventListener('click', async () => {
    const status = document.getElementById('status');
    status.style.color = '#aaa';
    status.textContent = 'Solving...';

    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        // Step 1: ask content script for the CAPTCHA's pixel data.
        const imgData = await chrome.tabs.sendMessage(tab.id, { action: 'getImageData' });
        if (!imgData.ok) throw new Error(imgData.error);

        // Step 2: run inference here in the popup (TF.js loaded as a normal script).
        status.textContent = 'Running model...';
        const result = await solveFromPixels(imgData.pixels);

        // Step 3: tell content script to fill the input field.
        await chrome.tabs.sendMessage(tab.id, { action: 'fillInput', text: result });

        status.textContent = `Result: ${result}`;
        status.style.color = '#2ecc71';

    } catch (err) {
        if (err.message.includes('Receiving end does not exist')) {
            status.textContent = 'Reload the page and try again.';
        } else {
            status.textContent = `Error: ${err.message}`;
        }
        status.style.color = '#e74c3c';
    }
});
