import joplin from 'api';
import { MenuItemLocation } from 'api/types';
import { ask_jarvis, chat_with_jarvis, edit_with_jarvis, research_with_jarvis } from './jarvis';
import { register_settings } from './settings';

joplin.plugins.register({
	onStart: async function() {
    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    register_settings();

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

    joplin.views.menus.create('jarvis', 'Jarvis', [
      {commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J'},
      {commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C'},
      {commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R'},
      {commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E'}
      ], MenuItemLocation.Tools
    );
	},
});
