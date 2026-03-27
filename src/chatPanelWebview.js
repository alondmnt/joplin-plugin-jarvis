(() => {
  let md = null;
  let markdownLoadPromise = null;
  const history = [];
  let initialized = false;
  let requestInFlight = false;
  let chatLog = null;
  let chatInput = null;
  let sendButton = null;
  let saveButton = null;
  let newButton = null;

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
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function tryInitMarkdown() {
    if (md) {
      return true;
    }
    if (typeof window.markdownit !== 'function') {
      return false;
    }
    md = window.markdownit({
      html: false,
      breaks: true,
      linkify: true,
      typographer: true,
    });
    return true;
  }

  function renderFallbackMarkdown(text) {
    const safe = escapeHtml(String(text));
    const linked = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|joplin:\/\/[^\s)]+)\)/g, '<a href="$2">$1</a>');
    const strong = linked
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>');
    const emphasized = strong
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/_(.+?)_/g, '<em>$1</em>');
    const coded = emphasized.replace(/`([^`]+)`/g, '<code>$1</code>');
    return coded
      .split(/\n{2,}/)
      .map((paragraph) => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
      .join('');
  }

  function renderMarkdown(text) {
    if (tryInitMarkdown()) {
      return {
        html: md.render(String(text).trim()),
        mode: 'markdown-it',
      };
    }
    return {
      html: renderFallbackMarkdown(text),
      mode: 'fallback',
    };
  }

  function rerenderFallbackMessages() {
    resolveElements();
    if (!chatLog || !tryInitMarkdown()) {
      return;
    }
    const messages = chatLog.querySelectorAll('.jarvis-chat-message[data-renderer="fallback"]');
    for (const body of messages) {
      const raw = body.getAttribute('data-raw') || '';
      body.innerHTML = md.render(raw.trim());
      body.setAttribute('data-renderer', 'markdown-it');
    }
  }

  function loadMarkdownIt() {
    if (tryInitMarkdown()) {
      return Promise.resolve(true);
    }
    if (markdownLoadPromise) {
      return markdownLoadPromise;
    }

    const sources = [
      'https://cdnjs.cloudflare.com/ajax/libs/markdown-it/13.0.2/markdown-it.min.js',
      'https://cdn.jsdelivr.net/npm/markdown-it@13.0.2/dist/markdown-it.min.js',
      'https://unpkg.com/markdown-it@13.0.2/dist/markdown-it.min.js',
    ];

    markdownLoadPromise = (async () => {
      for (const src of sources) {
        const alreadyLoaded = Array.from(document.getElementsByTagName('script')).some((script) => script.src === src);
        if (alreadyLoaded && tryInitMarkdown()) {
          rerenderFallbackMessages();
          return true;
        }

        const loaded = await new Promise((resolve) => {
          const script = document.createElement('script');
          script.src = src;
          script.async = true;
          script.onload = () => resolve(true);
          script.onerror = () => resolve(false);
          document.head.appendChild(script);
        });

        if (loaded && tryInitMarkdown()) {
          rerenderFallbackMessages();
          return true;
        }
      }

      return false;
    })().finally(() => {
      markdownLoadPromise = null;
    });

    return markdownLoadPromise;
  }

  function scrollToBottom() {
    resolveElements();
    if (!chatLog) {
      return;
    }
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendMessage(role, text) {
    resolveElements();
    if (!chatLog || !text) {
      return;
    }

    const row = document.createElement('div');
    row.className = role === 'assistant' ? 'jarvis-chat-row assistant' : 'jarvis-chat-row user';

    const label = document.createElement('div');
    label.className = 'jarvis-chat-role';
    label.textContent = role === 'assistant' ? 'Assistant' : 'User';

    const body = document.createElement('div');
    body.className = 'jarvis-chat-message';
    const rendered = renderMarkdown(text);
    body.innerHTML = rendered.html;
    body.setAttribute('data-raw', String(text));
    body.setAttribute('data-renderer', rendered.mode);

    row.appendChild(label);
    row.appendChild(body);
    chatLog.appendChild(row);
    scrollToBottom();
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
    history.push({ role: 'assistant', content: text });
    appendMessage('assistant', text);
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
    requestInFlight = true;
    setSending(true);

    try {
      const response = await withTimeout(webviewApi.postMessage({
        type: 'chatWithNotes',
        prompt,
        history,
      }), 120000);
      handleBackendMessage(response);
    } catch (error) {
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
    loadMarkdownIt();

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
      }
    });

    if (chatLog) {
      chatLog.addEventListener('click', (event) => {
        const target = event.target.closest('a');
        if (!target) return;
        const href = target.getAttribute('href') || '';
        if (href.startsWith('joplin://')) {
          event.preventDefault();
          const noteIdMatch = href.match(/id=([^&]+)/);
          if (noteIdMatch) {
            webviewApi.postMessage({ type: 'openNote', noteId: noteIdMatch[1] });
          }
        }
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
