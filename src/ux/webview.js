const CHAT_HISTORY_KEY = 'jarvis.panelChatHistory.v1';
const CHAT_DRAFT_KEY = 'jarvis.panelChatDraft.v1';

let panelChatHistory = loadChatHistory();

document.addEventListener('click', async event => {
  const element = event.target;

  if (element.tagName === 'A' && ((element.className === 'jarvis-semantic-section') || (element.className === 'jarvis-semantic-note'))) {
    webviewApi.postMessage({
      name: 'openRelatedNote',
      note: element.dataset.note,
      line: element.dataset.line,
    });
    return;
  }

  if (element.className === 'jarvis-cancel-button') {
    webviewApi.postMessage({ name: 'abortUpdate' });
    return;
  }

  if (element.className === 'jarvis-chat-send-button') {
    await sendPanelChat();
    return;
  }

  if (element.className === 'jarvis-chat-save-button') {
    await savePanelChatToNote();
    return;
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

document.addEventListener('input', event => {
  const element = event.target;
  if (element && element.id === 'jarvis-chat-input') {
    persistDraft(element.value || '');
  }
});

function loadChatHistory() {
  try {
    const raw = sessionStorage.getItem(CHAT_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(item =>
      item &&
      (item.role === 'user' || item.role === 'assistant') &&
      typeof item.content === 'string'
    );
  } catch (_) {
    return [];
  }
}

function persistChatHistory() {
  try {
    sessionStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(panelChatHistory));
  } catch (_) {
    // no-op
  }
}

function loadDraft() {
  try {
    return sessionStorage.getItem(CHAT_DRAFT_KEY) || '';
  } catch (_) {
    return '';
  }
}

function persistDraft(value) {
  try {
    sessionStorage.setItem(CHAT_DRAFT_KEY, value || '');
  } catch (_) {
    // no-op
  }
}

function clearDraft() {
  try {
    sessionStorage.removeItem(CHAT_DRAFT_KEY);
  } catch (_) {
    // no-op
  }
}

function appendChatMessage(role, text, persist = false) {
  const log = document.getElementById('jarvis-chat-log');
  if (!log) return;

  const msg = document.createElement('div');
  msg.className = `jarvis-chat-message ${role}`;

  let label = 'Jarvis';
  if (role === 'user') label = 'You';
  if (role === 'system') label = 'System';

  msg.textContent = `${label}: ${text}`;
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;

  if (persist && (role === 'user' || role === 'assistant')) {
    panelChatHistory.push({ role, content: text });
    persistChatHistory();
  }
}

function hydrateChatUi() {
  const log = document.getElementById('jarvis-chat-log');
  if (!log) return;

  if (log.dataset.hydrated !== '1') {
    log.innerHTML = '';
    for (const item of panelChatHistory) {
      appendChatMessage(item.role, item.content, false);
    }
    log.dataset.hydrated = '1';
  }

  const input = document.getElementById('jarvis-chat-input');
  if (input && !input.value) {
    input.value = loadDraft();
  }
}

async function sendPanelChat() {
  const input = document.getElementById('jarvis-chat-input');
  if (!input) return;

  const prompt = input.value.trim();
  if (!prompt) return;

  appendChatMessage('user', prompt, true);
  input.value = '';
  clearDraft();

  const response = await webviewApi.postMessage({
    name: 'chatWithNotes',
    prompt,
    history: panelChatHistory,
  });

  if (response && response.ok) {
    const answer = response.answer || '';
    appendChatMessage('assistant', answer, true);
  } else {
    appendChatMessage('assistant', `Error: ${(response && response.error) ? response.error : 'Unknown error'}`);
  }
}

async function savePanelChatToNote() {
  if (!panelChatHistory.length) {
    appendChatMessage('system', 'Nothing to save yet.');
    return;
  }

  const response = await webviewApi.postMessage({
    name: 'savePanelChatToNote',
    history: panelChatHistory,
  });

  if (response && response.ok) {
    appendChatMessage('system', `Saved to note: ${response.title}`);
  } else {
    appendChatMessage('system', `Save failed: ${(response && response.error) ? response.error : 'Unknown error'}`);
  }
}

const observer = new MutationObserver(() => {
  hydrateChatUi();
});

observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
});

hydrateChatUi();
