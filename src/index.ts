import joplin from 'api';
import { MenuItemLocation } from 'api/types';
import { ask_jarvis, chat_with_jarvis, edit_with_jarvis, find_notes, refresh_db, research_with_jarvis } from './jarvis';
import { get_settings, register_settings } from './settings';
import { load_model } from './embeddings';
import { connect_to_db, get_all_embeddings, init_db, clear_db } from './db';

joplin.plugins.register({
	onStart: async function() {
    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    await register_settings();

    await new Promise(res => setTimeout(res, 5 * 1000));
    const settings = await get_settings();
    const model = await load_model(settings);
    const db = await connect_to_db();
    await init_db(db);
    let embeddings = await get_all_embeddings(db);

    const panel = await joplin.views.panels.create('jarvis.relatedNotes');
    await joplin.views.panels.addScript(panel, './webview.css');
    await joplin.views.panels.addScript(panel, './webview.js');
    // TODO: move to an init_panel function
    await joplin.views.panels.setHtml(panel, '<div class="container"><p class="semantic-title">RELATED NOTES</p></div>');

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
      name: 'jarvis.notes.db.refresh',
      label: 'Refresh Jarvis note DB',
      execute: async () => {
        embeddings = await refresh_db(db, embeddings, model, panel);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.db.clear',
      label: 'Clear Jarvis note DB',
      execute: async () => {
        await clear_db(db);
      }
    });

    joplin.commands.register({
      name: 'jarvis.notes.find',
      label: 'Find related notes with Jarvis',
      execute: async () => {
        if (await joplin.views.panels.visible(panel)) {
          find_notes(panel, embeddings, model);
        }
      }
    });

    joplin.views.menus.create('jarvis', 'Jarvis', [
      {commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J'},
      {commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C'},
      {commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R'},
      {commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E'},
      {commandName: 'jarvis.notes.find', accelerator: 'CmdOrCtrl+Alt+F'}
      ], MenuItemLocation.Tools
    );

    joplin.views.menuItems.create('jarvis.notes.find', 'jarvis.notes.find', MenuItemLocation.EditorContextMenu);

    await joplin.views.panels.onMessage(panel, async (message) => {
      if (message.name === 'openRelatedNote') {
        await joplin.commands.execute('openNote', message.note);
        // Navigate to the line
        if (message.line > 0) {
          await new Promise(res => setTimeout(res, 1000));
          await joplin.commands.execute('editor.execCommand', {
            name: 'sidebar_cm_scrollToLine',
            args: [message.line - 1]
          });
        }
      }
    });
	},
});
