import joplin from 'api';
import { MenuItemLocation } from 'api/types';
import { ask_jarvis, chat_with_jarvis, edit_with_jarvis, embed_note, refresh_db, research_with_jarvis } from './jarvis';
import { get_settings, register_settings } from './settings';
import { BlockEmbedding, load_model } from './embeddings';
import { connect_to_db, get_all_embeddings, init_db } from './db';

joplin.plugins.register({
	onStart: async function() {
    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    await register_settings();

    // await new Promise(res => setTimeout(res, 10 * 1000));
    const settings = await get_settings();
    const model = await load_model(settings);
    const db = await connect_to_db();
    await init_db(db);
    let embeddings = await get_all_embeddings(db);

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
      name: 'jarvis.refreshDB',
      label: 'Refresh Jarvis DB',
      execute: async () => {
        embeddings = await refresh_db(db, embeddings, model);
      }
    });

    joplin.commands.register({
      name: 'jarvis.embed',
      label: 'Embed selection with Jarvis',
      execute: async () => {
        embed_note(embeddings, model);
      }
    });

    joplin.views.menus.create('jarvis', 'Jarvis', [
      {commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J'},
      {commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C'},
      {commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R'},
      {commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E'},
      {commandName: 'jarvis.embed', accelerator: 'CmdOrCtrl+Shift+I'}
      ], MenuItemLocation.Tools
    );
	},
});
