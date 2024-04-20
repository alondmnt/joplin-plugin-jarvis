import joplin from 'api';
import { ContentScriptType, MenuItemLocation, ToolbarButtonLocation } from 'api/types';
import * as debounce from 'lodash.debounce';
import { annotate_title, annotate_summary, annotate_tags, annotate_links } from './commands/annotate';
import { ask_jarvis, edit_with_jarvis } from './commands/ask';
import { chat_with_jarvis, chat_with_notes } from './commands/chat';
import { find_notes, update_note_db, skip_db_init_dialog } from './commands/notes';
import { research_with_jarvis } from './commands/research';
import { load_embedding_model, load_generation_model } from './models/models';
import { find_nearest_notes } from './notes/embeddings';
import { register_panel, update_panel } from './ux/panel';
import { get_settings, register_settings, set_folders } from './ux/settings';

joplin.plugins.register({
	onStart: async function() {
    await register_settings();
    let settings = await get_settings();

    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    const delay_startup = 5;  // seconds
    const delay_panel = 1;
    const delay_scroll = 1;
    let delay_db_update = 60 * settings.notes_db_update_delay;

    await new Promise(res => setTimeout(res, delay_startup * 1000));
    let model_embed = await load_embedding_model(settings);
    if (await skip_db_init_dialog(model_embed)) { delay_db_update = 0; }  // cancel auto update

    const panel = await joplin.views.panels.create('jarvis.relatedNotes');
    register_panel(panel, settings, model_embed);

    const find_notes_debounce = debounce(find_notes, delay_panel * 1000);
    if (model_embed.model) { find_notes_debounce(model_embed, panel) };
    let update_note_db_debounce = debounce(update_note_db, delay_db_update * 1000, {leading: true, trailing: false});

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
    })

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
        await update_note_db(model_embed, panel);
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

    await joplin.workspace.onNoteSelectionChange(async () => {
        if (model_embed.model === null) {
          await model_embed.initialize();
        }
        await find_notes_debounce(model_embed, panel);
        if (delay_db_update > 0) {
          await update_note_db_debounce(model_embed, panel);
        }
    });

    await joplin.views.panels.onMessage(panel, async (message) => {
      if (message.name === 'openRelatedNote') {
        await joplin.commands.execute('openNote', message.note);
        // Navigate to the line
        if (message.line > 0) {
          await new Promise(res => setTimeout(res, delay_scroll * 1000));
          await joplin.commands.execute('editor.execCommand', {
            name: 'scrollToJarvisLine',
            args: [message.line - 1]
          });
        }
      }
      if (message.name == 'searchRelatedNote') {
        const nearest = await find_nearest_notes(
          model_embed.embeddings, '1234', '', message.query, model_embed, settings);
        await update_panel(panel, nearest, settings);
      }
    });

    await joplin.settings.onChange(async (event) => {
      settings = await get_settings();
      // validate hugging face max tokens  
      if ((event.keys.includes('chat_hf_model_id') ||
           event.keys.includes('model') ||
           event.keys.includes('max_tokens')) &&
          (settings.model === 'Hugging Face') &&
          (settings.max_tokens > 2048)) {
        const choice = await joplin.views.dialogs.showMessageBox(
          `Hugging Face models typically do not exceed 2048 tokens, yet max tokens is currently set to ${settings.max_tokens}. Would you like to change it to 2048?`);
        if (choice === 0) {
          await joplin.settings.setValue('max_tokens', 2048);
          settings = await get_settings();
        }
      }
      // load generation model
      if (event.keys.includes('openai_api_key') ||
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
          event.keys.includes('notes_model') ||
          event.keys.includes('notes_parallel_jobs') ||
          event.keys.includes('notes_max_tokens') ||
          event.keys.includes('notes_openai_model_id') ||
          event.keys.includes('notes_openai_endpoint') ||
          event.keys.includes('notes_hf_model_id') ||
          event.keys.includes('notes_hf_endpoint')) {

        model_embed = await load_embedding_model(settings);
        if (model_embed.model) {
          await update_note_db(model_embed, panel);
        }
      }
      // update panel
      if (model_embed.model) {
        find_notes_debounce(model_embed, panel)
      };
      // update db refresh interval
      if (event.keys.includes('notes_db_update_delay')) {
        delay_db_update = 60 * settings.notes_db_update_delay;
        update_note_db_debounce = debounce(update_note_db,
          delay_db_update * 1000, {leading: true, trailing: false});
      }
    });
	},
});
