document.addEventListener('click', async event => {
	const element = event.target;
	// Only trigger navigation for actual <a> elements, not <summary> elements
	// This prevents the caret/arrow from opening notes on mobile/web
	if (element.tagName === 'A' && ((element.className === 'jarvis-semantic-section') || (element.className === 'jarvis-semantic-note'))) {
		// Post the message and slug info back to the plugin:
		webviewApi.postMessage({
			name: 'openRelatedNote',
			note: element.dataset.note,
			line: element.dataset.line,
		});
		return;
	}
	if (element.className === 'jarvis-cancel-button') {
		webviewApi.postMessage({
			name: 'abortUpdate'
		});
        return;
	}
   if (element.className === 'jarvis-chat-send-button') {
        await sendPanelChat();
    }
});

document.addEventListener('search', event => {
	const element = event.target;
	if (element.className === 'jarvis-semantic-query') {
		webviewApi.postMessage({
			name: 'searchRelatedNote',
			query: element.value,
		});
	}
});

document.addEventListener('keydown', async event => {
    if (event.key !== 'Enter' || event.shiftKey) return;
    const element = event.target;
    if (element && element.id === 'jarvis-chat-input') {
        event.preventDefault();
        await sendPanelChat();
    }
});

function appendChatMessage(role, text) {
    const log = document.getElementById('jarvis-chat-log');
    if (!log) return;

    const msg = document.createElement('div');
    msg.className = `jarvis-chat-message ${role}`;
    msg.textContent = `${role === 'user' ? 'You' : 'Jarvis'}: ${text}`;
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
}

async function sendPanelChat() {
    const input = document.getElementById('jarvis-chat-input');
    if (!input) return;

    const prompt = input.value.trim();
    if (!prompt) return;

    appendChatMessage('user', prompt);
    input.value = '';

    const response = await webviewApi.postMessage({
        name: 'chatWithNotes',
        prompt,
    });

    if (response && response.ok) {
        appendChatMessage('assistant', response.answer || '');
    } else {
        appendChatMessage('assistant', `Error: ${(response && response.error) ? response.error : 'Unknown error'}`);
    }
}
