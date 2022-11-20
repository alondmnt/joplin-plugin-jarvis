import joplin from 'api';
import { MenuItemLocation } from 'api/types';
import { ask_jarvis } from './jarvis';
import { register_settings } from './settings';


joplin.plugins.register({
	onStart: async function() {
    const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');

    register_settings();

    joplin.commands.register({
      name: 'jarvis.ask',
      label: 'Ask Jarvis',
      execute: async () => {
        ask_jarvis(dialogAsk)
      }
    });

    joplin.views.menuItems.create(
      'jarvis.ask', 'jarvis.ask', MenuItemLocation.Tools, {accelerator: 'CmdOrCtrl+Shift+J'});
	},
});
