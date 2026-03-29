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

  function renderInlineMarkdown(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+)__/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/_([^_]+)_/g, '<em>$1</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|joplin:\/\/[^\s)]+|:\/[^\s)]+)\)/g, '<a href="$2">$1</a>')
      .replace(/(^|[\s(>])((?:https?:\/\/|joplin:\/\/|:\/)[^\s<)]+)/g, '$1<a href="$2">$2</a>');
  }

  function renderCustomMarkdown(text) {
    const lines = String(text).replace(/\r\n/g, '\n').split('\n');
    const output = [];
    let inCodeBlock = false;
    let codeLines = [];
    let codeLanguage = '';
    let paragraph = [];
    let currentListType = '';

    const flushParagraph = () => {
      if (!paragraph.length) {
        return;
      }
      output.push(`<p>${paragraph.map((line) => renderInlineMarkdown(line)).join('<br>')}</p>`);
      paragraph = [];
    };

    const closeList = () => {
      if (!currentListType) {
        return;
      }
      output.push(`</${currentListType}>`);
      currentListType = '';
    };

    for (const rawLine of lines) {
      const line = rawLine || '';
      const trimmed = line.trim();

      const fence = trimmed.match(/^```\s*([a-zA-Z0-9_-]+)?\s*$/);
      if (fence) {
        flushParagraph();
        closeList();
        if (inCodeBlock) {
          const lang_attr = codeLanguage ? ` data-lang="${escapeHtml(codeLanguage)}"` : '';
          const lang_class = codeLanguage ? ` language-${escapeHtml(codeLanguage)}` : '';
          const code_text = escapeHtml(codeLines.join('\n'));
          output.push(`<pre class="jarvis-code-block${lang_class}"${lang_attr}><code>${code_text}</code></pre>`);
          inCodeBlock = false;
          codeLines = [];
          codeLanguage = '';
        } else {
          codeLanguage = fence[1] || '';
          inCodeBlock = true;
        }
        continue;
      }

      if (inCodeBlock) {
        codeLines.push(line);
        continue;
      }

      if (!trimmed) {
        flushParagraph();
        closeList();
        continue;
      }

      const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        closeList();
        const level = heading[1].length;
        output.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }

      const unordered = trimmed.match(/^[-*+]\s+(.+)$/);
      if (unordered) {
        flushParagraph();
        if (currentListType !== 'ul') {
          closeList();
          output.push('<ul>');
          currentListType = 'ul';
        }
        output.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
        continue;
      }

      const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
      if (ordered) {
        flushParagraph();
        if (currentListType !== 'ol') {
          closeList();
          output.push('<ol>');
          currentListType = 'ol';
        }
        output.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
        continue;
      }

      if (trimmed.startsWith('>')) {
        flushParagraph();
        closeList();
        output.push(`<blockquote>${renderInlineMarkdown(trimmed.slice(1).trim())}</blockquote>`);
        continue;
      }

      closeList();
      paragraph.push(line);
    }

    flushParagraph();
    closeList();
    if (inCodeBlock) {
      const lang_attr = codeLanguage ? ` data-lang="${escapeHtml(codeLanguage)}"` : '';
      const lang_class = codeLanguage ? ` language-${escapeHtml(codeLanguage)}` : '';
      const code_text = escapeHtml(codeLines.join('\n'));
      output.push(`<pre class="jarvis-code-block${lang_class}"${lang_attr}><code>${code_text}</code></pre>`);
    }

    return output.join('');
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
    body.innerHTML = renderCustomMarkdown(text);

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
        type: useNotes ? 'chatWithNotes' : 'chat',
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
