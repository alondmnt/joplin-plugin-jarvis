import joplin from 'api';
import { find_nearest_notes, update_embeddings } from '../notes/embeddings';
import { update_panel, update_progress_bar } from '../ux/panel';
import { get_settings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';


export async function update_note_db(model: TextEmbeddingModel, panel: string, abortController: AbortController): Promise<void> {
  if (model.model === null) { return; }

  const settings = await get_settings();

  let notes: any;
  let page = 0;
  let total_notes = 0;
  let processed_notes = 0;

  // count all notes
  do {
    page += 1;
    notes = await joplin.data.get(['notes'], { fields: ['id'], page: page });
    total_notes += notes.items.length;
  } while (notes.has_more);
  update_progress_bar(panel, 0, total_notes, settings);

  page = 0;
  // iterate over all notes
  do {
    page += 1;
    notes = await joplin.data.get(['notes'], { fields: ['id', 'title', 'body', 'is_conflict', 'parent_id', 'deleted_time', 'markup_language'], page: page, limit: model.page_size });
    if (notes.items) {
      console.log(`Processing page ${page}: ${notes.items.length} notes`);
      await update_embeddings(notes.items, model, settings, abortController);
      processed_notes += notes.items.length;
      update_progress_bar(panel, processed_notes, total_notes, settings);
    }
    // rate limiter
    if (notes.has_more && (page % model.page_cycle) == 0) {
      console.log(`Waiting for ${model.wait_period} seconds...`);
      await new Promise(res => setTimeout(res, model.wait_period * 1000));
    }
  } while (notes.has_more);

  find_notes(model, panel);
}

export async function find_notes(model: TextEmbeddingModel, panel: string) {
  if (!(await joplin.views.panels.visible(panel))) {
    return;
  }
  if (model.model === null) { return; }
  const settings = await get_settings();

  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }
  if (note.markup_language === 2) {
    return;
  }
  let selected = await joplin.commands.execute('selectedText');
  if (!selected || (selected.length === 0)) {
    selected = note.body;
  }
  const nearest = await find_nearest_notes(model.embeddings, note.id, note.title, selected, model, settings);

  // write results to panel
  await update_panel(panel, nearest, settings);
}

export async function skip_db_init_dialog(model: TextEmbeddingModel): Promise<boolean> {
  if (model.embeddings.length > 0) { return false; }

  let calc_msg = `This database is calculated locally (offline) by running ${model.id}`;
  let compute = 'PC';
  if (model.online) {
    calc_msg = `This database is calculated remotely (online) by sending requests to ${model.id}`;
    compute = 'connection';
  }
  return (await joplin.views.dialogs.showMessageBox(
    `Hi! Jarvis can build a database of your notes, that may be used to search for similar notes, or to chat with your notes.
    
    ${calc_msg}, and then stored in a local sqlite database.
    
    *If* you choose to chat with your notes, short excerpts from the database will be send to an online/offline model of your choosing.
    
    You can delete the database at any time by deleting the file. Initialization may take between a few minutes (fast ${compute}, ~500 notes collection) and a couple of hours.
    
    Press 'OK' to run it now in the background, or 'Cancel' to postpone it to a later time (e.g., overnight). You can start the process at any time from Tools-->Jarvis-->Update Jarvis note DB. You may delay it indefinitely by setting the 'Database update period' to 0.`
    ) == 1);
}
