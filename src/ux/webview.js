document.addEventListener('click', event => {
	const element = event.target;
	if ((element.className === 'jarvis-semantic-section') || (element.className === 'jarvis-semantic-note')) {
		// Post the message and slug info back to the plugin:
		webviewApi.postMessage({
			name: 'openRelatedNote',
			note: element.dataset.note,
			line: element.dataset.line,
		});
	}
	if (element.className === 'jarvis-cancel-button') {
		webviewApi.postMessage({
			name: 'abortUpdate'
		});
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
