(() => {
  const history = [];
  let initialized = false;
  let requestInFlight = false;
  let useNotes = true;
  let chatLog = null;
  let chatInput = null;
  let sendButton = null;
  let saveButton = null;
  let newButton = null;
  let modeButton = null;
  let draftTimer = null;

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

  function setSending(isSending) {
    resolveElements();
    if (sendButton) {
      sendButton.disabled = isSending;
      sendButton.textContent = isSending ? 'Sending...' : 'Send';
    }
    if (chatInput) {
      chatInput.disabled = isSending;
    }
  }

  function withTimeout(promise, ms) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('Request timed out.'));
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
    if (typeof message.useNotes === 'boolean') {
      useNotes = message.useNotes;
      if (modeButton) modeButton.textContent = useNotes ? 'Notes' : 'Chat';
      if (chatInput) chatInput.placeholder = useNotes ? 'Ask Jarvis about your notes...' : 'Chat with Jarvis...';
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
      // Backend caught a failure and surfaced it as text. Don't pollute
      // history with the error or the failed user turn — show the error
      // in the DOM as a transient message and restore the prompt to the
      // input so the user can retry without retyping.
      let restored = null;
      if (history.length > 0 && history[history.length - 1].role === 'user') {
        restored = history.pop().content;
      }
      if (restored !== null && chatInput && !chatInput.value.trim()) {
        chatInput.value = restored;
        webviewApi.postMessage({ type: 'draftChange', draft: restored });
      }
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
        type: useNotes ? 'chatWithNotes' : 'chat',
        prompt,
        history,
      }), 120000);
      removeThinking(thinking);
      handleBackendMessage(response);
    } catch (error) {
      removeThinking(thinking);
      // Roll back the optimistic user push so a failed turn doesn't leave
      // two consecutive user roles in history (which strict chat templates
      // such as Gemma reject on the next send).
      if (history.length > 0 && history[history.length - 1].role === 'user') {
        history.pop();
      }
      // Restore the typed prompt so the user can retry without retyping —
      // unless they already started typing the next prompt while waiting.
      if (chatInput && !chatInput.value.trim()) {
        chatInput.value = prompt;
        webviewApi.postMessage({ type: 'draftChange', draft: prompt });
      }
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
      }), 30000);
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
        sendPrompt();
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
        useNotes = !useNotes;
        target.textContent = useNotes ? 'Notes' : 'Chat';
        resolveElements();
        if (chatInput) {
          chatInput.placeholder = useNotes ? 'Ask Jarvis about your notes...' : 'Chat with Jarvis...';
        }
        webviewApi.postMessage({ type: 'modeChange', useNotes });
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
        useNotes = !useNotes;
        resolveElements();
        if (modeButton) modeButton.textContent = useNotes ? 'Notes' : 'Chat';
        if (chatInput) chatInput.placeholder = useNotes ? 'Ask Jarvis about your notes...' : 'Chat with Jarvis...';
        webviewApi.postMessage({ type: 'modeChange', useNotes });
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
