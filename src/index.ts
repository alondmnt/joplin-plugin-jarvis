import joplin from 'api';
import { MenuItemLocation } from 'api/types';
import * as debounce from 'lodash.debounce';
import { ask_jarvis, chat_with_jarvis, edit_with_jarvis, find_notes, update_note_db, research_with_jarvis, chat_with_notes } from './jarvis';
import { get_settings, register_settings } from './settings';
import { load_model } from './embeddings';
import { connect_to_db, get_all_embeddings, init_db } from './db';
import { register_panel } from './panel';

joplin.plugins.register({
	onStart: async function() {
    await register_settings();
    const settings = await get_settings();

    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    const delay_startup = 5;  // seconds
    const delay_panel = 1;
    const delay_scroll = 1;
    const delay_db_update = 60 * settings.notes_db_update_delay;

    await new Promise(res => setTimeout(res, delay_startup * 1000));
    let model = await load_model(settings);
    const db = await connect_to_db();
    await init_db(db);
    let embeddings = await get_all_embeddings(db);

    const panel = await joplin.views.panels.create('jarvis.relatedNotes');
    register_panel(panel, settings, model);

    const find_notes_debounce = debounce(find_notes, delay_panel * 1000);
    const update_note_db_debounce = debounce(update_note_db, delay_db_update * 1000, {leading: true, trailing: false});

    joplin.commands.register({
      name: 'jarvis.ask',
      label: 'Ask Jarvis',
      execute: async () => {
        ask_jarvis(dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.chat',
      label: 'Chat with Jarvis',
      execute: async () => {
        chat_with_jarvis();
      }
    })

    joplin.commands.register({
      name: 'jarvis.research',
      label: 'Research with Jarvis',
      execute: async () => {
        research_with_jarvis(dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.edit',
      label: 'Edit selection with Jarvis',
      execute: async () => {
        edit_with_jarvis(dialogAsk);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.db.update',
      label: 'Update Jarvis note DB',
      execute: async () => {
        if (model === null) {
          model = await load_model(settings);
        }
        embeddings = await update_note_db(db, embeddings, model, panel);
        find_notes_debounce(panel, embeddings, model);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.find',
      label: 'Find related notes',
      execute: async () => {
        if (model === null) {
          model = await load_model(settings);
        }
        find_notes_debounce(panel, embeddings, model);
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
          if (model === null) {
            model = await load_model(settings);
          }
          find_notes_debounce(panel, embeddings, model)
        }
      },
    });

    joplin.commands.register({
      name: 'jarvis.notes.chat',
      label: 'Chat with your notes',
      execute: async () => {
        if (model === null) {
          model = await load_model(settings);
        }
        chat_with_notes(embeddings, model);
      }
    });

    joplin.views.menus.create('jarvis', 'Jarvis', [
      {commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C'},
      {commandName: 'jarvis.notes.chat', accelerator: 'CmdOrCtrl+Alt+C'},
      {commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J'},
      {commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R'},
      {commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E'},
      {commandName: 'jarvis.notes.find', accelerator: 'CmdOrCtrl+Alt+F'},
      {commandName: 'jarvis.notes.db.update'},
      {commandName: 'jarvis.notes.toggle_panel'},
      ], MenuItemLocation.Tools
    );

    joplin.views.menuItems.create('jarvis.notes.find', 'jarvis.notes.find', MenuItemLocation.EditorContextMenu);

    await joplin.workspace.onNoteSelectionChange(async () => {
        if (model === null) {
          model = await load_model(settings);
        }
        await find_notes_debounce(panel, embeddings, model);
        if (delay_db_update > 0) {
          embeddings = await update_note_db_debounce(db, embeddings, model, panel);
        }
    });

    await joplin.views.panels.onMessage(panel, async (message) => {
      if (message.name === 'openRelatedNote') {
        await joplin.commands.execute('openNote', message.note);
        // Navigate to the line
        if (message.line > 0) {
          await new Promise(res => setTimeout(res, delay_scroll * 1000));
          await joplin.commands.execute('editor.execCommand', {
            name: 'sidebar_cm_scrollToLine',
            args: [message.line - 1]
          });
        }
      }
    });
	},
});
