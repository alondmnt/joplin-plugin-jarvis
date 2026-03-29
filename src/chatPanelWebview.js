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

  function extractNoteId(href) {
    if (!href || typeof href !== 'string') {
      return '';
    }

    if (href.startsWith(':/')) {
      const m = href.match(/^:\/([^?#&]+)/);
      return m ? m[1] : '';
    }

    if (href.startsWith('joplin://')) {
      const m = href.match(/[?&]id=([^&]+)/);
      return m ? decodeURIComponent(m[1]) : '';
    }

    return '';
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
        if (chatInput) chatInput.focus();
        return;
      }
      if (target.id === 'chat-mode') {
        useNotes = !useNotes;
        target.textContent = useNotes ? 'Notes' : 'Chat';
        resolveElements();
        if (chatInput) {
          chatInput.placeholder = useNotes ? 'Ask Jarvis about your notes...' : 'Chat with Jarvis...';
        }
      }
    });

    if (chatLog) {
      chatLog.addEventListener('click', (event) => {
        const target = event.target.closest('a');
        if (!target) return;
        const href = target.getAttribute('href') || '';
        if (href.startsWith('joplin://') || href.startsWith(':/')) {
          event.preventDefault();
          const noteId = extractNoteId(href);
          if (noteId) {
            webviewApi.postMessage({ type: 'openNote', noteId });
          }
        }
      });
    }

    if (chatInput) {
      chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
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
    });

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();
