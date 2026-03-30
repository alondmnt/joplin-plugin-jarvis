let loading_active = false;
let loading_observer = null;

function get_message_name(message) {
	if (!message) {
		return '';
	}
	if (typeof message === 'string') {
		return message;
	}
	if (typeof message.name === 'string') {
		return message.name;
	}
	if (message.message && typeof message.message.name === 'string') {
		return message.message.name;
	}
	return '';
}

function ensure_loading_element() {
	let el = document.getElementById('jarvis-loading');
	if (el) {
		return el;
	}

	const container = document.querySelector('.container');
	if (!container) {
		return null;
	}

	el = document.createElement('div');
	el.id = 'jarvis-loading';
	el.className = 'jarvis-loading';
	el.innerHTML = '<div class="jarvis-spinner"></div><span>Jarvis is working...</span>';

	const title = container.querySelector('.jarvis-semantic-title');
	if (title && title.nextSibling) {
		container.insertBefore(el, title.nextSibling);
	} else if (title) {
		title.insertAdjacentElement('afterend', el);
	} else {
		container.prepend(el);
	}

	return el;
}

function sync_loading_state() {
	const el = ensure_loading_element();
	if (!el) {
		return;
	}
	if (loading_active) {
		el.classList.add('active');
		// Inline fallback in case class-based CSS is not applied yet.
		el.style.display = 'flex';
	} else {
		el.classList.remove('active');
		el.style.display = 'none';
	}
}

function ensure_loading_observer() {
	if (loading_observer) {
		return;
	}
	loading_observer = new MutationObserver(() => {
		if (loading_active) {
			sync_loading_state();
		}
	});
	loading_observer.observe(document.documentElement, { childList: true, subtree: true });
}

function show_loading() {
	loading_active = true;
	sync_loading_state();
}

function hide_loading() {
	loading_active = false;
	sync_loading_state();
}

webviewApi.onMessage((message) => {
	const name = get_message_name(message);
	if (name === 'show_loading') {
		show_loading();
	}
	if (name === 'hide_loading') {
		hide_loading();
	}
});

document.addEventListener('DOMContentLoaded', () => {
	ensure_loading_observer();
	sync_loading_state();
});

document.addEventListener('click', event => {
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
