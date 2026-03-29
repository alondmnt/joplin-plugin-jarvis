import joplin from 'api';
import type { TextEmbeddingModel, TextGenerationModel } from './models/models';
import type { JarvisSettings } from './ux/settings';
import { chat_with_notes_panel, format_as_note_chat, type PanelChatMessage } from './commands/chat';
import { clearObjectReferences } from './utils';

type ChatPanelContext = {
  model_embed: TextEmbeddingModel;
  model_gen: TextGenerationModel;
  settings: JarvisSettings;
};

function sanitize_history(history: unknown): PanelChatMessage[] {
  if (!Array.isArray(history)) {
    return [];
  }

  return history
    .filter((entry) => entry && typeof entry === 'object')
    .map((entry: any) => {
      const role: PanelChatMessage['role'] = entry.role === 'assistant' ? 'assistant' : 'user';
      return {
        role,
        content: typeof entry.content === 'string' ? entry.content : '',
      };
    })
    .filter((entry) => entry.content.trim().length > 0);
}


async function resolve_parent_notebook_id(): Promise<string> {
  const selected_note = await joplin.workspace.selectedNote();
  try {
    if (selected_note?.parent_id) {
      return selected_note.parent_id;
    }
  } finally {
    clearObjectReferences(selected_note);
  }

  const selected_folder = await joplin.workspace.selectedFolder();
  if (selected_folder?.id) {
    return selected_folder.id;
  }

  const folder_page: any = await joplin.data.get(['folders'], { fields: ['id'], limit: 1 });
  if (folder_page?.items?.length > 0 && folder_page.items[0]?.id) {
    return folder_page.items[0].id;
  }

  throw new Error('No target notebook is available to save this chat.');
}

export async function initialize_chat_panel(get_context: () => ChatPanelContext): Promise<string> {
  const panel = await joplin.views.panels.create('jarvis_chat_panel');
  await joplin.views.panels.addScript(panel, 'chatPanel.css');
  await joplin.views.panels.addScript(panel, 'chatPanelWebview.js');
  await joplin.views.panels.setHtml(panel, `
  <div class="jarvis-chat-panel">
<div id="chat-log" class="jarvis-chat-log" aria-live="polite"></div>
    <div class="jarvis-chat-input-wrap">
      <span id="chat-mode" class="jarvis-chat-mode">Notes</span>
      <textarea id="chat-input" class="jarvis-chat-input" placeholder="Ask Jarvis about your notes..." rows="2"></textarea>
    </div>
    <div class="jarvis-chat-actions">
      <button id="chat-send" type="button">Send</button>
      <button id="chat-save" type="button">Save to Note</button>
      <button id="chat-new" type="button">New Chat</button>
    </div>
  </div>
  `);

  await joplin.views.panels.onMessage(panel, async (message: any) => {
    if (!message || typeof message !== 'object') {
      return { type: 'response', text: 'Invalid panel message.' };
    }

    if (message.type === 'chatWithNotes') {
      const prompt = typeof message.prompt === 'string' ? message.prompt.trim() : '';
      if (!prompt) {
        return { type: 'response', text: 'Please enter a prompt.' };
      }

      const runtime = get_context();
      if (runtime.model_gen?.model === null && typeof runtime.model_gen?.initialize === 'function') {
        await runtime.model_gen.initialize();
      }
      if (runtime.model_embed?.model === null && typeof runtime.model_embed?.initialize === 'function') {
        await runtime.model_embed.initialize();
      }
      if (!runtime.model_gen?.model || !runtime.model_embed?.model) {
        return { type: 'response', text: 'Jarvis models are not initialized yet. Please try again in a moment.' };
      }

      try {
        const history = sanitize_history(message.history);
        const text = await chat_with_notes_panel(
          prompt,
          history,
          runtime.model_embed,
          runtime.model_gen,
          runtime.settings,
        );
        return { type: 'response', text };
      } catch (error) {
        const msg = error instanceof Error ? error.message : 'Unknown error';
        return { type: 'response', text: `Chat failed: ${msg}` };
      }
    }

    if (message.type === 'chat') {
      const prompt = typeof message.prompt === 'string' ? message.prompt.trim() : '';
      if (!prompt) {
        return { type: 'response', text: 'Please enter a prompt.' };
      }

      const runtime = get_context();
      if (runtime.model_gen?.model === null && typeof runtime.model_gen?.initialize === 'function') {
        await runtime.model_gen.initialize();
      }
      if (!runtime.model_gen?.model) {
        return { type: 'response', text: 'Jarvis model is not initialised yet. Please try again in a moment.' };
      }

      try {
        const history = sanitize_history(message.history);
        const full_prompt = format_as_note_chat(history, runtime.settings);

        const raw = await runtime.model_gen.chat(full_prompt);
        const text = (raw || '')
          .replace(runtime.model_gen.model_prefix, '')
          .replace(runtime.model_gen.user_prefix, '')
          .trim();
        return { type: 'response', text };
      } catch (error) {
        const msg = error instanceof Error ? error.message : 'Unknown error';
        return { type: 'response', text: `Chat failed: ${msg}` };
      }
    }

    if (message.type === 'savePanelChatToNote') {
      const history = sanitize_history(message.history);
      if (history.length === 0) {
        return { type: 'saved', text: 'Nothing to save yet.' };
      }

      try {
        const runtime = get_context();
        const parent_id = await resolve_parent_notebook_id();
        const title_stamp = new Date().toISOString().replace('T', ' ').replace('Z', ' UTC');
        const note = await joplin.data.post(['notes'], null, {
          title: `Jarvis Chat ${title_stamp}`,
          body: format_as_note_chat(history, runtime.settings),
          parent_id,
        });
        return { type: 'saved', text: `Saved to note: ${note.title}` };
      } catch (error) {
        const msg = error instanceof Error ? error.message : 'Unknown error';
        return { type: 'saved', text: `Save failed: ${msg}` };
      }
    }

    if (message.type === 'openNote') {
      const noteId = typeof message.noteId === 'string' ? message.noteId.trim() : '';
      if (noteId) {
        await joplin.commands.execute('openNote', noteId);
      }
      return { type: 'ack' };
    }

    return { type: 'response', text: 'Unsupported panel message type.' };
  });

  return panel;
}
