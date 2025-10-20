import joplin from 'api';
import { ContentScriptType, MenuItemLocation, ToolbarButtonLocation } from 'api/types';
import debounce from 'lodash.debounce';
import { annotate_title, annotate_summary, annotate_tags, annotate_links } from './commands/annotate';
import { ask_jarvis, edit_with_jarvis } from './commands/ask';
import { chat_with_jarvis, chat_with_notes } from './commands/chat';
import { find_notes, update_note_db, skip_db_init_dialog } from './commands/notes';
import { research_with_jarvis } from './commands/research';
import { load_embedding_model, load_generation_model } from './models/models';
import { find_nearest_notes } from './notes/embeddings';
import { register_panel, update_panel } from './ux/panel';
import { get_settings, register_settings, set_folders } from './ux/settings';
import { auto_complete } from './commands/complete';

joplin.plugins.register({
	onStart: async function() {
    await register_settings();
    let settings = await get_settings();

    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    const delay_startup = 5;  // seconds
    const delay_panel = 1;
    let delay_scroll = await joplin.settings.value('notes_scroll_delay');
    const abort_timeout = 10;  // minutes
    let delay_db_update = 60 * settings.notes_db_update_delay;

    await new Promise(res => setTimeout(res, delay_startup * 1000));
    let model_embed = await load_embedding_model(settings);
    if (await skip_db_init_dialog(model_embed)) { delay_db_update = 0; }  // cancel auto update

    // Track in-progress updates so we can merge overlapping requests and avoid UI stalls.
    let updateAbortController: AbortController | null = null;
    let updateStartTime: number | null = null;
    const pendingNoteIds = new Set<string>();  // deduplicated queue of notes awaiting rebuild
    const noteChangeDebounceMs = 2000;

    interface UpdateOptions {
      force?: boolean;
      noteIds?: string[];
      silent?: boolean;
    }

    // Helper function to check if update is in progress
    function is_update_in_progress(): boolean {
      console.debug('is_update_in_progress', updateAbortController, updateStartTime);
      return updateAbortController !== null &&
        (updateStartTime !== null && (Date.now() - updateStartTime) < abort_timeout * 60 * 1000);
    }

    /**
     * Kick off an embedding rebuild, optionally scoped to a set of note IDs.
     * Collapses concurrent callers, reusing the shared abort controller when forced.
     * Note the legacy snake_case helpers (`update_note_db`, `find_notes`) this wraps.
     */
    async function start_update(model_embed: any, panel: string, options: UpdateOptions = {}) {
      const { force = false, noteIds, silent = false } = options;
      const targetIds = noteIds && noteIds.length > 0 ? Array.from(new Set(noteIds)) : undefined;
      console.debug('start_update', is_update_in_progress(), force, targetIds?.length ?? 0);

      if (targetIds && targetIds.length === 0 && !force) {
        return;
      }

      if (is_update_in_progress()) {
        if (!force) {
          if (!silent) {
            await joplin.views.dialogs.showMessageBox('Update already in progress');
          }
          return;
        }
        if (updateAbortController) {
          updateAbortController.abort();
        }
      } else if (updateAbortController !== null) {
        updateAbortController.abort();
      }

      if (updateAbortController !== null) {
        // ensure previous controller can't leak to new updates
        updateAbortController = null;
      }
      updateAbortController = new AbortController();
      updateStartTime = Date.now();

      try {
        await update_note_db(model_embed, panel, updateAbortController, targetIds);
      } finally {
        updateAbortController = null;
        updateStartTime = null;
      }
    }

    const panel = await joplin.views.panels.create('jarvis.relatedNotes');
    register_panel(panel, settings, model_embed);

    const find_notes_debounce = debounce(find_notes, delay_panel * 1000);
    if (model_embed.model) { find_notes_debounce(model_embed, panel) };
    let update_note_db_debounce = debounce(async (model_embed: any, panel: string) => {
      await start_update(model_embed, panel);
    }, delay_db_update * 1000, {leading: true, trailing: false});

    let model_gen = await load_generation_model(settings);

    await joplin.contentScripts.register(
      ContentScriptType.CodeMirrorPlugin,
      'jarvis.cm5scroller',
      './content_scripts/cm5scroller.js',
    );
    await joplin.contentScripts.register(
      ContentScriptType.CodeMirrorPlugin,
      'jarvis.cm6scroller',
      './content_scripts/cm6scroller.js',
    );

    joplin.commands.register({
      name: 'jarvis.ask',
      label: 'Ask Jarvis',
      execute: async () => {
        ask_jarvis(model_gen, dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.chat',
      label: 'Chat with Jarvis',
      iconName: 'fas fa-robot',
      execute: async () => {
        chat_with_jarvis(model_gen);
      }
    });

    joplin.commands.register({
      name: 'jarvis.research',
      label: 'Research with Jarvis',
      execute: async () => {
        research_with_jarvis(model_gen, dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.edit',
      label: 'Edit selection with Jarvis',
      iconName: 'far fa-edit',
      execute: async () => {
        edit_with_jarvis(model_gen, dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.complete',
      label: 'Auto-complete with Jarvis',
      execute: async () => {
        auto_complete(model_gen, settings);
      }
    });

    joplin.commands.register({
      name: 'jarvis.annotate.title',
      label: 'Annotate note: title',
      execute: async () => {
        await annotate_title(model_gen, settings);
      }
    });

    joplin.commands.register({
      name: 'jarvis.annotate.summary',
      label: 'Annotate note: summary',
      execute: async () => {
        await annotate_summary(model_gen, settings);
      }
    });

    joplin.commands.register({
      name: 'jarvis.annotate.tags',
      label: 'Annotate note: tags',
      execute: async () => {
        await annotate_tags(model_gen, model_embed, settings);
      }
    });

    joplin.commands.register({
      name: 'jarvis.annotate.links',
      label: 'Annotate note: links',
      execute: async () => {
        await annotate_links(model_embed, settings);
      }
    });

    joplin.commands.register({
      name: 'jarvis.annotate.button',
      label: 'Annotate note with Jarvis',
      iconName: 'fas fa-lightbulb',
      execute: async () => {
        if (settings.annotate_links_flag) { await annotate_links(model_embed, settings); }

        if (settings.annotate_summary_flag || settings.annotate_title_flag || settings.annotate_tags_flag) {
          // use a single big prompt to generate a summary, and then reuse it for title and tags
          const summary = await annotate_summary(model_gen, settings, settings.annotate_summary_flag);
          if (settings.annotate_title_flag) { await annotate_title(model_gen, settings, summary); }
          if (settings.annotate_tags_flag) { await annotate_tags(model_gen, model_embed, settings, summary); }
          }
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.db.update',
      label: 'Update Jarvis note DB',
      execute: async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        await start_update(model_embed, panel);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.find',
      label: 'Find related notes',
      iconName: 'fas fa-search',
      execute: async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        find_notes_debounce(model_embed, panel);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.toggle_panel',
      label: 'Toggle related notes panel',
      execute: async () => {
        if (await joplin.views.panels.visible(panel)) {
          await joplin.views.panels.hide(panel);
        } else {
          await joplin.views.panels.show(panel);
          if (model_embed.model === null) {
            await model_embed.initialize();
          }
          find_notes_debounce(model_embed, panel)
        }
      },
    });

    joplin.commands.register({
      name: 'jarvis.notes.chat',
      label: 'Chat with your notes',
      iconName: 'fas fa-comments',
      execute: async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        chat_with_notes(model_embed, model_gen, panel);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.preview',
      label: 'Preview chat notes context',
      execute: async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        chat_with_notes(model_embed, model_gen, panel, true);
      }
    });

    joplin.commands.register({
      name: 'jarvis.utils.count_tokens',
      label: 'Count tokens in selection',
      execute: async () => {
        const text = await joplin.commands.execute('selectedText');
        const token_count = model_gen.count_tokens(text);
        await joplin.views.dialogs.showMessageBox(`Token count: ${token_count}`);
      },
    });

    await joplin.commands.register({
      name: 'jarvis.notes.exclude_folder',
      label: 'Exclude notebook from note DB',
      execute: async () => {
        const folder = await joplin.workspace.selectedFolder();
        if (folder == undefined) return;

        set_folders(true, folder.id, settings);
      },
    });

    await joplin.commands.register({
      name: 'jarvis.notes.include_folder',
      label: 'Include notebook in note DB',
      execute: async () => {
        const folder = await joplin.workspace.selectedFolder();
        if (folder == undefined) return;

        set_folders(false, folder.id, settings);
      },
    });

    joplin.views.menus.create('jarvis', 'Jarvis', [
      {commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C'},
      {commandName: 'jarvis.notes.chat', accelerator: 'CmdOrCtrl+Alt+C'},
      {commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J'},
      {commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R'},
      {commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E'},
      {commandName: 'jarvis.complete', accelerator: 'CmdOrCtrl+Shift+A'},
      {commandName: 'jarvis.annotate.title'},
      {commandName: 'jarvis.annotate.summary'},
      {commandName: 'jarvis.annotate.links'},
      {commandName: 'jarvis.annotate.tags'},
      {commandName: 'jarvis.notes.find', accelerator: 'CmdOrCtrl+Alt+F'},
      {commandName: 'jarvis.notes.preview'},
      {commandName: 'jarvis.utils.count_tokens'},
      {commandName: 'jarvis.notes.db.update'},
      {commandName: 'jarvis.notes.toggle_panel'},
      {commandName: 'jarvis.notes.exclude_folder'},
      {commandName: 'jarvis.notes.include_folder'},
      ], MenuItemLocation.Tools
    );

    joplin.views.toolbarButtons.create('jarvis.toolbar.notes.find', 'jarvis.notes.find', ToolbarButtonLocation.EditorToolbar);
    joplin.views.toolbarButtons.create('jarvis.toolbar.edit', 'jarvis.edit', ToolbarButtonLocation.EditorToolbar);
    joplin.views.toolbarButtons.create('jarvis.toolbar.chat', 'jarvis.chat', ToolbarButtonLocation.EditorToolbar);
    joplin.views.toolbarButtons.create('jarvis.toolbar.annotate', 'jarvis.annotate.button', ToolbarButtonLocation.EditorToolbar);

    joplin.views.menuItems.create('jarvis.context.notes.find', 'jarvis.notes.find', MenuItemLocation.EditorContextMenu);
    joplin.views.menuItems.create('jarvis.context.utils.count_tokens', 'jarvis.utils.count_tokens', MenuItemLocation.EditorContextMenu);
    joplin.views.menuItems.create('jarvis.context.edit', 'jarvis.edit', MenuItemLocation.EditorContextMenu);

    /**
     * Debounced worker that drains pending note IDs into the background updater.
     * Keeps compatibility with snake_case workhorses by calling `start_update`.
     */
    const flush_note_changes = debounce(async () => {
      if (pendingNoteIds.size === 0) {
        return;
      }

      if (is_update_in_progress()) {
        flush_note_changes();
        return;
      }

      if (model_embed.model === null) {
        flush_note_changes();
        return;
      }

      const noteIds = Array.from(pendingNoteIds);
      pendingNoteIds.clear();
      try {
        // Silent mode avoids spurious dialogs when rapid edits trigger overlapping batches.
        await start_update(model_embed, panel, { noteIds, silent: true });
      } catch (error) {
        // Requeue on failure so the next pass can retry after a transient error.
        for (const id of noteIds) {
          pendingNoteIds.add(id);
        }
        throw error;
      }
    }, noteChangeDebounceMs);

    await joplin.workspace.onNoteSelectionChange(async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        await find_notes_debounce(model_embed, panel);
        if (delay_db_update > 0) {
          await update_note_db_debounce(model_embed, panel);
        }
    });

    await joplin.workspace.onNoteChange(async (event: any) => {
      const noteId = event?.id ?? event?.item?.id;
      if (!noteId) {
        return;
      }
      pendingNoteIds.add(noteId);
      flush_note_changes();
    });

    await joplin.views.panels.onMessage(panel, async (message) => {
      if (message.name === 'openRelatedNote') {
        await joplin.commands.execute('openNote', message.note);
        // Navigate to the line
        if (message.line > 0) {
          await new Promise(res => setTimeout(res, delay_scroll));
          await joplin.commands.execute('editor.execCommand', {
            name: 'scrollToJarvisLine',
            args: [message.line - 1]
          });
        }
      }
      if (message.name == 'searchRelatedNote') {
        const nearest = await find_nearest_notes(
          model_embed.embeddings, '1234', 1, '', message.query, model_embed, settings);
        await update_panel(panel, nearest, settings);
      }
      if (message.name === 'abortUpdate') {
        if (updateAbortController) {
          updateAbortController.abort();
        }
      }
    });

    await joplin.settings.onChange(async (event) => {
      settings = await get_settings();
      // load generation model
      if (event.keys.includes('openai_api_key') ||
          event.keys.includes('anthropic_api_key') ||
          event.keys.includes('hf_api_key') ||
          event.keys.includes('google_api_key') ||
          event.keys.includes('model') ||
          event.keys.includes('chat_system_message') ||
          event.keys.includes('chat_timeout') ||
          event.keys.includes('chat_openai_model_id') ||
          event.keys.includes('chat_openai_model_type') ||
          event.keys.includes('chat_openai_endpoint') ||
          event.keys.includes('max_tokens') ||
          event.keys.includes('memory_tokens') ||
          event.keys.includes('notes_context_tokens') ||
          event.keys.includes('temperature') ||
          event.keys.includes('top_p') ||
          event.keys.includes('frequency_penalty') ||
          event.keys.includes('presence_penalty') ||
          event.keys.includes('chat_hf_model_id') ||
          event.keys.includes('chat_hf_endpoint') ||
          event.keys.includes('chat_prefix') ||
          event.keys.includes('chat_suffix')) {

        model_gen = await load_generation_model(settings);
      }
      // load embedding model
      if (event.keys.includes('openai_api_key') ||
          event.keys.includes('hf_api_key') ||
          event.keys.includes('google_api_key') ||
          event.keys.includes('notes_model') ||
          event.keys.includes('notes_embed_title') ||
          event.keys.includes('notes_embed_path') ||
          event.keys.includes('notes_embed_heading') ||
          event.keys.includes('notes_embed_tags') ||
          event.keys.includes('notes_parallel_jobs') ||
          event.keys.includes('notes_max_tokens') ||
          event.keys.includes('notes_openai_model_id') ||
          event.keys.includes('notes_openai_endpoint') ||
          event.keys.includes('notes_hf_model_id') ||
          event.keys.includes('notes_hf_endpoint') ||
          event.keys.includes('notes_abort_on_error') ||
          event.keys.includes('notes_embed_timeout')) {

        model_embed = await load_embedding_model(settings);
        if (model_embed.model) {
          await start_update(model_embed, panel, { force: true });
        }
      }
      if (event.keys.includes('notes_scroll_delay')) {
        delay_scroll = await joplin.settings.value('notes_scroll_delay');
      }
      // update panel
      if (model_embed.model) {
        find_notes_debounce(model_embed, panel);
      };
      // update db refresh interval
      if (event.keys.includes('notes_db_update_delay')) {
        delay_db_update = 60 * settings.notes_db_update_delay;
        update_note_db_debounce = debounce(async (model_embed: any, panel: string) => {
          await start_update(model_embed, panel);
        }, delay_db_update * 1000, {leading: true, trailing: false});
      }
    });
	},
});
