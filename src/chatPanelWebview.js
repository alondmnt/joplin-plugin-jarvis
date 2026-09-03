(() => {
  const MODES = {
    chat:       { label: 'Chat',       placeholder: 'Chat with Jarvis...',          messageType: 'chat' },
    note:       { label: 'Note',       placeholder: 'Ask about the current note...', messageType: 'chatWithNote' },
    collection: { label: 'Collection', placeholder: 'Ask Jarvis about your notes...', messageType: 'chatWithNotes' },
  };
  // Rotation order: collection (default) → note → chat → collection. Click and
  // Shift+Tab both advance through this cycle.
  const MODE_ORDER = ['collection', 'note', 'chat'];
  const nextMode = (mode) => MODE_ORDER[(MODE_ORDER.indexOf(mode) + 1) % MODE_ORDER.length];

  // Liveness backstops, not deadlines. The deadline belongs to the model, which
  // applies "Chat: Timeout (sec)" per request; these only exist so a backend
  // that never answers can't leave the panel disabled forever.
  //
  // Sized to clear the slowest run that completes on its own. A collection
  // query is a whole pipeline rather than one request (query embedding, a
  // similarity scan over the corpus, an optional query-decomposition
  // completion, then the answer), so two model requests at up to 600s each
  // plus the scan. Anything derived from the per-request timeout would fire
  // mid-query on a large corpus.
  //
  // One path still exceeds it: on a model timeout, timeout_with_retry offers
  // an interactive retry and recurses with no cap, so each OK adds another
  // request. Bounding that means letting the panel opt out of the modal, which
  // is a signature change on the model's chat() and belongs in its own change.
  //
  // Save only writes a note, so it needs far less room.
  const SEND_LIVENESS_MS = 30 * 60 * 1000;
  const SAVE_LIVENESS_MS = 2 * 60 * 1000;

  const history = [];
  let initialized = false;
  let requestInFlight = false;
  let chatMode = 'collection';
  let chatLog = null;
  let chatInput = null;
  let sendButton = null;
  let saveButton = null;
  let newButton = null;
  let modeButton = null;
  let draftTimer = null;

  function applyMode(mode) {
    resolveElements();
    const cfg = MODES[mode] || MODES.collection;
    if (modeButton) modeButton.textContent = cfg.label;
    if (chatInput) chatInput.placeholder = cfg.placeholder;
  }

  function resolveElements() {
    if (!chatLog) {
      chatLog = document.getElementById('chat-log');
    }
    if (!chatInput) {
      chatInput = document.getElementById('chat-input');
    }
    if (!sendButton) {
      sendButton = document.getElementById('chat-send');
    }
    if (!saveButton) {
      saveButton = document.getElementById('chat-save');
    }
    if (!newButton) {
      newButton = document.getElementById('chat-new');
    }
    if (!modeButton) {
      modeButton = document.getElementById('chat-mode');
    }
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function scrollToBottom() {
    resolveElements();
    if (!chatLog) {
      return;
    }
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendMessage(role, content, html) {
    resolveElements();
    if (!chatLog || !content) {
      return;
    }

    const row = document.createElement('div');
    row.className = role === 'assistant' ? 'jarvis-chat-row assistant' : 'jarvis-chat-row user';

    const body = document.createElement('div');
    body.className = 'jarvis-chat-message';
    body.innerHTML = html || escapeHtml(content).replace(/\n/g, '<br>');

    row.appendChild(body);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function showThinking() {
    resolveElements();
    if (!chatLog) return null;
    const row = document.createElement('div');
    row.className = 'jarvis-chat-row assistant';
    row.innerHTML = '<div class="jarvis-thinking"><span>.</span><span>.</span><span>.</span></div>';
    chatLog.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeThinking(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  // Undo an optimistic user turn: pop it from history, remove its DOM
  // bubble, and restore the typed prompt to the input (unless the user
  // started typing the next prompt while waiting). Bails out if there
  // is no trailing user turn to roll back, so the DOM and history can't
  // drift if an exception fires after an assistant entry was added.
  function rollbackFailedTurn() {
    resolveElements();
    if (history.length === 0 || history[history.length - 1].role !== 'user') {
      return;
    }
    const restored = history.pop().content;
    if (chatLog) {
      const userRows = chatLog.querySelectorAll('.jarvis-chat-row.user');
      const lastUser = userRows[userRows.length - 1];
      if (lastUser && lastUser.parentNode) {
        lastUser.parentNode.removeChild(lastUser);
      }
    }
    if (chatInput && !chatInput.value.trim()) {
      chatInput.value = restored;
      webviewApi.postMessage({ type: 'draftChange', draft: restored });
    }
  }

  // Send doubles as Stop while a request is running: the button has to stay
  // enabled for that, so the guard against a second send is requestInFlight in
  // the click handler rather than a disabled button.
  function setSending(isSending) {
    resolveElements();
    if (sendButton) {
      sendButton.disabled = false;
      sendButton.textContent = isSending ? 'Stop' : 'Send';
    }
    if (chatInput) {
      chatInput.disabled = isSending;
    }
  }

  function stopPrompt() {
    if (!requestInFlight) {
      return;
    }
    webviewApi.postMessage({ type: 'cancelChat' });
  }

  function withTimeout(promise, ms) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        // No action to advise: reloading the panel repaints from the backend
        // cache mid-request and New Chat leaves the reply to land on an empty
        // one, because neither cancels the request. Report the state instead.
        reject(new Error('Jarvis stopped responding. The request may still be running in the background.'));
      }, ms);

      Promise.resolve(promise)
        .then((value) => {
          clearTimeout(timer);
          resolve(value);
        })
        .catch((error) => {
          clearTimeout(timer);
          reject(error);
        });
    });
  }

  function restoreState(message) {
    if (!message || typeof message !== 'object' || message.type !== 'restore') return;

    resolveElements();

    // apply mobile layout adjustment
    if (message.platform === 'mobile') {
      document.querySelector('.jarvis-chat-panel')?.classList.add('mobile');
    }

    // restore mode
    if (typeof message.chatMode === 'string' && MODES[message.chatMode]) {
      chatMode = message.chatMode;
      applyMode(chatMode);
    }

    // restore draft
    if (typeof message.draft === 'string' && message.draft && chatInput) {
      chatInput.value = message.draft;
      chatInput.style.height = 'auto';
      chatInput.style.height = chatInput.scrollHeight + 'px';
    }

    // restore chat history
    if (Array.isArray(message.history) && message.history.length > 0) {
      history.length = 0;
      if (chatLog) chatLog.innerHTML = '';
      for (const entry of message.history) {
        if (!entry || typeof entry !== 'object') continue;
        const role = entry.role === 'assistant' ? 'assistant' : 'user';
        const content = typeof entry.content === 'string' ? entry.content : '';
        const html = typeof entry.html === 'string' ? entry.html : '';
        if (!content) continue;
        history.push({ role, content });
        appendMessage(role, content, html);
      }
    }
  }

  function handleBackendMessage(message) {
    if (!message || typeof message !== 'object') {
      appendMessage('assistant', 'Received an invalid response from Jarvis.');
      return;
    }

    if (message.type === 'saved') {
      const text = typeof message.text === 'string' ? message.text : 'Chat saved.';
      appendMessage('assistant', text);
      return;
    }

    const text = typeof message.text === 'string' ? message.text : '';
    const html = typeof message.html === 'string' ? message.html : '';

    if (message.error === true) {
      // Backend caught a failure and surfaced it as text. Roll back the
      // failed user turn so it doesn't pollute history or the visible
      // panel, then show the error as a transient assistant message.
      rollbackFailedTurn();
      appendMessage('assistant', text, html);
      return;
    }

    history.push({ role: 'assistant', content: text });
    appendMessage('assistant', text, html);
  }

  async function sendPrompt() {
    resolveElements();
    if (!chatInput || requestInFlight) {
      return;
    }

    const prompt = chatInput.value.trim();
    if (!prompt) {
      return;
    }

    history.push({ role: 'user', content: prompt });
    appendMessage('user', prompt);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    clearTimeout(draftTimer);
    webviewApi.postMessage({ type: 'draftChange', draft: '' });
    requestInFlight = true;
    setSending(true);
    const thinking = showThinking();

    try {
      const response = await withTimeout(webviewApi.postMessage({
        type: (MODES[chatMode] || MODES.collection).messageType,
        prompt,
        history,
      }), SEND_LIVENESS_MS);
      removeThinking(thinking);
      handleBackendMessage(response);
    } catch (error) {
      removeThinking(thinking);
      rollbackFailedTurn();
      const message = error instanceof Error ? error.message : 'Unknown error';
      appendMessage('assistant', `Chat failed: ${message}`);
    } finally {
      requestInFlight = false;
      setSending(false);
      if (chatInput) {
        chatInput.focus();
      }
    }
  }

  async function saveChat() {
    resolveElements();
    try {
      const response = await withTimeout(webviewApi.postMessage({
        type: 'savePanelChatToNote',
        history,
      }), SAVE_LIVENESS_MS);
      handleBackendMessage(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      appendMessage('assistant', `Save failed: ${message}`);
    }
  }

  function setup() {
    if (initialized) {
      return;
    }
    initialized = true;
    resolveElements();

    document.addEventListener('click', (event) => {
      const target = event.target;
      if (!target || typeof target.id !== 'string') {
        return;
      }
      if (target.id === 'chat-send') {
        if (requestInFlight) {
          stopPrompt();
        } else {
          sendPrompt();
        }
        return;
      }
      if (target.id === 'chat-save') {
        saveChat();
        return;
      }
      if (target.id === 'chat-new') {
        history.length = 0;
        resolveElements();
        if (chatLog) chatLog.innerHTML = '';
        if (chatInput) { chatInput.value = ''; chatInput.focus(); }
        webviewApi.postMessage({ type: 'newChat' });
        return;
      }
      if (target.id === 'chat-mode') {
        chatMode = nextMode(chatMode);
        applyMode(chatMode);
        webviewApi.postMessage({ type: 'modeChange', chatMode });
      }
    });

    if (chatLog) {
      chatLog.addEventListener('click', (event) => {
        const target = event.target.closest('a');
        if (!target) return;
        event.preventDefault();
        const href = target.getAttribute('href') || '';
        if (href) {
          webviewApi.postMessage({ type: 'openNote', href });
        }
      });
    }

    if (chatInput) {
      chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
        clearTimeout(draftTimer);
        draftTimer = setTimeout(() => {
          webviewApi.postMessage({ type: 'draftChange', draft: chatInput.value });
        }, 500);
      });
    }

    document.addEventListener('keydown', (event) => {
      const target = event.target;
      if (!target || target.id !== 'chat-input') {
        return;
      }
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendPrompt();
      }
      if (event.key === 'Tab' && event.shiftKey) {
        event.preventDefault();
        chatMode = nextMode(chatMode);
        applyMode(chatMode);
        webviewApi.postMessage({ type: 'modeChange', chatMode });
      }
    });

    // request cached state from plugin process
    webviewApi.postMessage({ type: 'initPanel' }).then(restoreState);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();
