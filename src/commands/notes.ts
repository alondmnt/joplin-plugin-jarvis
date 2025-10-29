import joplin from 'api';
import { find_nearest_notes, update_embeddings, build_centroid_index_on_startup } from '../notes/embeddings';
import { ensure_catalog_note, ensure_model_anchor, get_catalog_note_id } from '../notes/catalog';
import { update_panel, update_progress_bar } from '../ux/panel';
import { get_settings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { ModelError } from '../utils';


/**
 * Refresh embeddings for either the entire notebook set or a specific list of notes.
 * When `noteIds` are provided the function reuses the existing update pipeline to
 * rebuild only those entries, skipping the legacy full-library scan.
 */
export async function update_note_db(
  model: TextEmbeddingModel,
  panel: string,
  abortController: AbortController,
  noteIds?: string[],
): Promise<void> {
  if (model.model === null) { return; }

  const settings = await get_settings();

  // Ensure catalog and model anchor once before processing notes to prevent
  // concurrent per-note creation races when the experimental userData index is enabled.
  if (settings.experimental_user_data_index && model?.id) {
    try {
      const catalogId = await ensure_catalog_note();
      await ensure_model_anchor(catalogId, model.id, model.version ?? 'unknown');
      
      // Build centroid-to-note index during startup/full database update (§4 in task list)
      // Piggybacks on note scanning to populate index without additional overhead
      await build_centroid_index_on_startup(model.id, settings);
    } catch (error) {
      const msg = String((error as any)?.message ?? error);
      if (!/SQLITE_CONSTRAINT|UNIQUE constraint failed/i.test(msg)) {
        console.debug('Jarvis: deferred ensure anchor (update_note_db)', msg);
      }
    }
  }

  const noteFields = ['id', 'title', 'body', 'is_conflict', 'parent_id', 'deleted_time', 'markup_language'];
  let total_notes = 0;
  let processed_notes = 0;

  if (noteIds && noteIds.length > 0) {
    const uniqueIds = Array.from(new Set(noteIds));  // dedupe in case of repeated change events
    total_notes = uniqueIds.length;
    update_progress_bar(panel, 0, total_notes, settings);

    const batch: any[] = [];
    for (const noteId of uniqueIds) {
      if (abortController.signal.aborted) {
        break;
      }
      try {
        const note = await joplin.data.get(['notes', noteId], { fields: noteFields });
        if (note) {
          batch.push(note);
        }
      } catch (error) {
        console.debug(`Skipping note ${noteId}:`, error);
      }

      while (batch.length >= model.page_size && !abortController.signal.aborted) {
        // Flush in fixed-size chunks so we keep consistent throughput with the full scan path.
        const chunk = batch.splice(0, model.page_size);
        await update_embeddings(chunk, model, settings, abortController);
        processed_notes += chunk.length;
        update_progress_bar(panel, Math.min(processed_notes, total_notes), total_notes, settings);
      }
    }

    if (batch.length > 0 && !abortController.signal.aborted) {
      await update_embeddings(batch, model, settings, abortController);
      processed_notes += batch.length;
      update_progress_bar(panel, Math.min(processed_notes, total_notes), total_notes, settings);
    }
  } else {
    let notes: any;
    let page = 0;

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
      notes = await joplin.data.get(['notes'], { fields: noteFields, page: page, limit: model.page_size });
      if (notes.items) {
        console.debug(`Processing page ${page}: ${notes.items.length} notes`);
        await update_embeddings(notes.items, model, settings, abortController);
        processed_notes += notes.items.length;
        update_progress_bar(panel, processed_notes, total_notes, settings);
      }
      // rate limiter
      if (notes.has_more && (page % model.page_cycle) == 0) {
        console.debug(`Waiting for ${model.wait_period} seconds...`);
        await new Promise(res => setTimeout(res, model.wait_period * 1000));
      }
    } while (notes.has_more);
  }

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
  let selected = await joplin.commands.execute('selectedText');
  if (!selected || (selected.length === 0)) {
    selected = note.body;
  }
  let nearest;
  try {
    nearest = await find_nearest_notes(model.embeddings, note.id, note.markup_language, note.title, selected, model, settings);
  } catch (error) {
    if (error instanceof ModelError) {
      await joplin.views.dialogs.showMessageBox(`Error: ${error.message}`);
      return;
    }
    throw error;
  }

  // write results to panel
  await update_panel(panel, nearest, settings);
}

export async function skip_db_init_dialog(model: TextEmbeddingModel): Promise<boolean> {
  // Check if database has been initialized before:
  // 1. Legacy approach: embeddings loaded in memory
  if (model.embeddings.length > 0) { 
    return false; 
  }

  // 2. New approach: catalog note exists (works on both desktop and mobile)
  try {
    const catalogId = await get_catalog_note_id();
    if (catalogId) {
      // Database previously initialized, don't show welcome dialog
      return false;
    }
  } catch (error) {
    console.debug('Jarvis: catalog check during init failed', error);
  }

  // No database found - show welcome dialog
  const settings = await get_settings();
  const storageMethod = settings.experimental_user_data_index 
    ? 'stored as note attachments in your Joplin database (experimental)'
    : 'stored in a local SQLite database file';
  
  let calc_msg = `Embeddings are calculated locally (offline) by running ${model.id}`;
  let compute = 'PC';
  if (model.online) {
    calc_msg = `Embeddings are calculated remotely (online) by sending requests to ${model.id}`;
    compute = 'connection';
  }

  return (await joplin.views.dialogs.showMessageBox(
    `Welcome to Jarvis Note Search!

Jarvis can build a searchable database of your notes to help you:
  • Find similar notes based on meaning, not just keywords
  • Chat with your notes using AI that understands your content

HOW IT WORKS:
${calc_msg}, then ${storageMethod}. All data stays under your control.

PRIVACY:
When you chat with your notes, only relevant excerpts are sent to your chosen AI model. The database itself never leaves your device.

SETUP TIME:
Initial indexing takes ${compute === 'PC' ? '5-30 minutes' : '10-60 minutes'} for ~500 notes, depending on your ${compute}.

───────────────────────────────

Press OK to build the database now in the background.
Press Cancel to postpone (you can start anytime from Tools → Jarvis → Update Jarvis note DB).

Tip: You can disable automatic updates by setting 'Database update period' to 0 in settings.`
    ) == 1);
}
