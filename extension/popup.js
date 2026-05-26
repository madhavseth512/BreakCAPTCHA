'use strict';

document.getElementById('solveBtn').addEventListener('click', async () => {
    const status = document.getElementById('status');
    status.style.color = '#aaa';
    status.textContent = 'Solving...';

    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const response = await chrome.tabs.sendMessage(tab.id, { action: 'solve' });

        if (response.ok) {
            status.textContent = `Result: ${response.result}`;
            status.style.color = '#2ecc71';
        } else {
            status.textContent = `Error: ${response.error}`;
            status.style.color = '#e74c3c';
        }
    } catch (err) {
        // Most common cause: content scripts not yet injected (reload the tab).
        if (err.message.includes('Receiving end does not exist')) {
            status.textContent = 'Reload the page and try again.';
        } else {
            status.textContent = `Error: ${err.message}`;
        }
        status.style.color = '#e74c3c';
    }
});
