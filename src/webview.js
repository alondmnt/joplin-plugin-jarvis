document.addEventListener('click', event => {
	const element = event.target;
	if ((element.className === 'semantic-section') || (element.className === 'semantic-note')) {
		// Post the message and slug info back to the plugin:
		webviewApi.postMessage({
			name: 'openRelatedNote',
			note: element.dataset.note,
			line: element.dataset.line,
		});
	}
});
