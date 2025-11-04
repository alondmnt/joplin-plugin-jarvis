import joplin from 'api';
import { find_nearest_notes, update_embeddings, build_centroid_index_on_startup } from '../notes/embeddings';
import { ensure_catalog_note, ensure_model_anchor, get_catalog_note_id } from '../notes/catalog';
import { update_panel, update_progress_bar } from '../ux/panel';
import { get_settings, mark_model_first_build_completed, get_model_last_sweep_time, set_model_last_sweep_time } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { ModelError } from '../utils';
import { assign_missing_centroids } from '../notes/centroidAssignment';
import { UserDataEmbStore } from '../notes/userDataStore';
import type { JarvisSettings } from '../ux/settings';

/**
 * Refresh embeddings for either the entire notebook set or a specific list of notes.
 * When `noteIds` are provided the function reuses the existing update pipeline to
 * rebuild only those entries, skipping the legacy full-library scan.
 * 
 * @param force - Controls rebuild behavior:
 *   - false: Skip notes where content unchanged (only checks contentHash)
 *     Example: Note saved by user, incremental periodic sweeps
 *   - true: Skip only if content unchanged AND settings match AND model matches
 *     Example: Manual "Update DB", settings changed, validation dialog rebuild
 * 
 * @param incrementalSweep - When true, use timestamp-based early termination:
 *   - Query notes ordered by updated_time DESC
 *   - Stop when reaching notes older than model.lastIncrementalSweepTime
 *   - Typical cost: ~50-100ms for 10-50 updated notes
 *   - Used for: periodic background sweeps to catch sync changes
 *   - Requires: force=false (thoroughness incompatible with early termination)
 * 
 * Smart rebuild: Both modes skip when up-to-date, but "up-to-date" means different things:
 *   - force=false: Content unchanged (backfills userData from SQLite if needed for migration)
 *   - force=true: Content unchanged AND userData matches current settings/model/version
 * 
 * Multi-device benefit: If Device A syncs embeddings with new settings, Device B's force=true
 * rebuild will skip already-updated notes, saving API quota.
 */
export async function update_note_db(
  model: TextEmbeddingModel,
  panel: string,
  abortController: AbortController,
  noteIds?: string[],
  force: boolean = false,
  incrementalSweep: boolean = false,
): Promise<void> {
  if (model.model === null) { return; }

  const settings = await get_settings();
  
  // Log update start with last sweep timestamp for debugging
  const lastSweepTime = await get_model_last_sweep_time(model.id);
  const lastSweepDate = lastSweepTime > 0 ? new Date(lastSweepTime).toISOString() : 'never';
  const mode = noteIds && noteIds.length > 0 
    ? `specific notes (${noteIds.length})` 
    : incrementalSweep && !force 
      ? 'incremental sweep' 
      : 'full sweep';
  console.info(`Jarvis: starting update (mode: ${mode}, force: ${force}, last sweep: ${lastSweepDate}, incrementalSweep: ${incrementalSweep})`);


  // Ensure catalog and model anchor once before processing notes to prevent
  // concurrent per-note creation races when the experimental userData index is enabled.
  let catalogId: string | undefined;
  let anchorId: string | undefined;
  if (settings.experimental_user_data_index && model?.id) {
    try {
      catalogId = await ensure_catalog_note();
      anchorId = await ensure_model_anchor(catalogId, model.id, model.version ?? 'unknown');
      
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
  const allSettingsMismatches: Array<{ noteId: string; currentSettings: any; storedSettings: any }> = [];

  const isFullSweep = !noteIds || noteIds.length === 0;

  if (noteIds && noteIds.length > 0) {
    // Mode 1: Specific note IDs (note saves, sync notifications)
    const uniqueIds = Array.from(new Set(noteIds));  // dedupe in case of repeated change events
    total_notes = uniqueIds.length;
    update_progress_bar(panel, 0, total_notes, settings, 'Computing embeddings');

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

      // Flush in fixed-size chunks for consistent throughput
      while (batch.length >= model.page_size && !abortController.signal.aborted) {
        const chunk = batch.splice(0, model.page_size);
        const mismatches = await process_batch_and_update_progress(
          chunk, model, settings, abortController, force, catalogId, anchorId, panel,
          () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
          (count) => { processed_notes += count; }
        );
        allSettingsMismatches.push(...mismatches);
      }
    }

    // Process remaining notes
    if (!abortController.signal.aborted) {
      const mismatches = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, anchorId, panel,
        () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
        (count) => { processed_notes += count; }
      );
      allSettingsMismatches.push(...mismatches);
    }
  } else if (isFullSweep && incrementalSweep && !force) {
    // Mode 2: Timestamp-based incremental sweep (optimized for periodic background updates)
    // Uses user_updated_time (user content changes) instead of updated_time (system changes including userData writes)
    // This prevents reprocessing notes where only embeddings were written, while still catching synced content changes
    let page = 1;
    let hasMore = true;
    let reachedOldNotes = false;
    
    while (hasMore && !reachedOldNotes && !abortController.signal.aborted) {
      const notes = await joplin.data.get(['notes'], {
        fields: [...noteFields, 'user_updated_time'],
        page,
        limit: model.page_size,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      
      const batch: any[] = [];
      for (const note of notes.items) {
        // Stop when we reach notes older than last sweep (comparing user content timestamp)
        if (note.user_updated_time <= lastSweepTime) {
          reachedOldNotes = true;
          console.debug(`Reached old notes at page ${page}, stopping early`);
          break;
        }
        batch.push(note);
      }
      
      const mismatches = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, anchorId, panel,
        () => ({ processed: processed_notes, total: total_notes }),
        (count) => { processed_notes += count; total_notes += count; }  // Adjust estimate as we go
      );
      allSettingsMismatches.push(...mismatches);
      
      hasMore = notes.has_more && !reachedOldNotes;
      page++;
      
      await apply_rate_limit_if_needed(hasMore, page, model);
    }
    
    console.info(`Jarvis: incremental sweep completed - ${processed_notes} notes processed (${reachedOldNotes ? 'stopped early' : 'reached end of notes'})`);
  } else {
    // Mode 3: Full sweep (thorough scan of all notes)
    // Used for: manual rebuild, force=true validation, first-time builds
    let notes: any;
    let page = 0;

    // Count all notes for accurate progress bar
    do {
      page += 1;
      notes = await joplin.data.get(['notes'], { fields: ['id'], page: page });
      total_notes += notes.items.length;
    } while (notes.has_more);
    update_progress_bar(panel, 0, total_notes, settings, 'Computing embeddings');

    page = 0;
    // Iterate over all notes
    do {
      page += 1;
      notes = await joplin.data.get(['notes'], { fields: noteFields, page: page, limit: model.page_size });
      if (notes.items) {
        console.debug(`Processing page ${page}: ${notes.items.length} notes, total so far: ${processed_notes}/${total_notes}`);
        const mismatches = await process_batch_and_update_progress(
          notes.items, model, settings, abortController, force, catalogId, anchorId, panel,
          () => ({ processed: processed_notes, total: total_notes }),
          (count) => { processed_notes += count; }
        );
        allSettingsMismatches.push(...mismatches);
      }
      
      await apply_rate_limit_if_needed(notes.has_more, page, model);
    } while (notes.has_more);
    
    console.info(`Jarvis: full sweep completed - ${processed_notes}/${total_notes} notes processed successfully`);
    if (processed_notes < total_notes) {
      console.warn(`Jarvis: ${total_notes - processed_notes} notes were skipped or failed during update`);
    }
  }

  // Update last sweep timestamp after successful completion of any full sweep
  // (both incremental and thorough scans, but not specific note IDs)
  if (!abortController.signal.aborted && isFullSweep) {
    const timestamp = Date.now();
    await set_model_last_sweep_time(model.id, timestamp);
    console.debug(`Jarvis: updated last sweep time to ${new Date(timestamp).toISOString()}`);
  }

  // Check if centroids were trained/retrained during this sweep and assign centroid IDs immediately
  // This is critical for search correctness - missing or stale centroid IDs break IVF routing
  // This handles both first build (if corpus ≥512 rows) and subsequent retraining
  if (
    !abortController.signal.aborted
    && settings.experimental_user_data_index
    && model?.id
    && model.needsCentroidReassignment
  ) {
    console.info('Jarvis: centroids were trained/retrained, assigning centroid IDs to all notes', { modelId: model.id });
    try {
      const store = new UserDataEmbStore();
      const result = await assign_missing_centroids({
        model,
        store,
        abortSignal: abortController.signal,
        forceReassign: true, // Force reassignment even for notes with existing centroid IDs (handles stale assignments)
        onProgress: (processed, total) => {
          if (total > 0) {
            update_progress_bar(panel, processed, total, settings, 'Assigning centroid IDs');
          }
        },
      });
      console.info('Jarvis: centroid assignment completed', {
        modelId: model.id,
        ...result,
      });
      // Clear the flag so we don't reassign again unless centroids are retrained
      model.needsCentroidReassignment = false;
    } catch (error) {
      console.warn('Jarvis: centroid assignment failed', {
        modelId: model.id,
        error: String((error as any)?.message ?? error),
      });
      // Don't clear the flag - we'll try again next time
      // Notes without centroid IDs will fall back to brute-force search (functional but slower)
    }
  }

  // Mark first build as complete after main sweep finishes
  // This is separate from centroid assignment - even if corpus is small (<512 rows) and no centroids exist,
  // we still mark first build complete so we stop loading legacy SQLite database
  if (
    !abortController.signal.aborted
    && isFullSweep
    && settings.experimental_user_data_index
    && model?.id
    && !settings.notes_model_first_build_completed?.[model.id]
  ) {
    // Close legacy SQLite database
    if (model.db && typeof model.db.close === 'function') {
      try {
        await new Promise<void>((resolve, reject) => {
          model.db.close((error: any) => {
            if (error) {
              reject(error);
            } else {
              resolve();
            }
          });
        });
      } catch (error) {
        console.warn('Jarvis: failed to close legacy SQLite database after first build', error);
      }
    }
    model.db = null;
    model.disableDbLoad = true;
    
    await mark_model_first_build_completed(model.id);
    console.info('Jarvis: first userData build completed, disabled legacy SQLite access', { modelId: model.id });
  }

  // Show settings mismatch dialog if mismatches were found during sweep (only for force=false sweeps)
  if (!force && allSettingsMismatches.length > 0 && settings.experimental_user_data_index) {
    // Deduplicate by noteId
    const uniqueMismatches = Array.from(
      new Map(allSettingsMismatches.map(m => [m.noteId, m])).values()
    );
    
    // Format human-readable summary of settings changes
    const settingsDiffs: string[] = [];
    for (const mismatch of uniqueMismatches.slice(0, 1)) { // Check first mismatch as representative
      const current = mismatch.currentSettings;
      const stored = mismatch.storedSettings;
      
      if (current.embedTitle !== stored.embedTitle) {
        settingsDiffs.push(`embedTitle ${stored.embedTitle ? 'Yes' : 'No'}→${current.embedTitle ? 'Yes' : 'No'}`);
      }
      if (current.embedPath !== stored.embedPath) {
        settingsDiffs.push(`embedPath ${stored.embedPath ? 'Yes' : 'No'}→${current.embedPath ? 'Yes' : 'No'}`);
      }
      if (current.embedHeading !== stored.embedHeading) {
        settingsDiffs.push(`embedHeading ${stored.embedHeading ? 'Yes' : 'No'}→${current.embedHeading ? 'Yes' : 'No'}`);
      }
      if (current.embedTags !== stored.embedTags) {
        settingsDiffs.push(`embedTags ${stored.embedTags ? 'Yes' : 'No'}→${current.embedTags ? 'Yes' : 'No'}`);
      }
      if (current.includeCode !== stored.includeCode) {
        settingsDiffs.push(`includeCode ${stored.includeCode ? 'Yes' : 'No'}→${current.includeCode ? 'Yes' : 'No'}`);
      }
      if (current.maxTokens !== stored.maxTokens) {
        settingsDiffs.push(`maxTokens ${stored.maxTokens}→${current.maxTokens}`);
      }
    }
    
    const message = `Found ${uniqueMismatches.length} note(s) with different embedding settings (likely synced from another device).\n\nSettings: ${settingsDiffs.join(', ')}\n\nRebuild these notes with current settings?`;
    
    const choice = await joplin.views.dialogs.showMessageBox(message);
    
    if (choice === 0) {
      // User chose to rebuild
      console.info(`Jarvis: User chose to rebuild ${uniqueMismatches.length} notes with mismatched settings`);
      // Trigger full scan with force=true to rebuild all mismatched notes
      await joplin.commands.execute('jarvis.notes.db.update');
      return; // Don't call find_notes, the rebuild will do it
    } else {
      // User chose to skip
      console.info(`Jarvis: User declined to rebuild ${uniqueMismatches.length} notes with mismatched settings`);
    }
  }

  find_notes(model, panel);
}

/**
 * Process a batch of notes and update progress bar.
 * Common helper to avoid duplication across sweep modes.
 * Returns settings mismatches found during processing.
 */
async function process_batch_and_update_progress(
  batch: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
  force: boolean,
  catalogId: string | undefined,
  anchorId: string | undefined,
  panel: string,
  getProgress: () => { processed: number; total: number },
  onProcessed: (count: number) => void,
): Promise<Array<{ noteId: string; currentSettings: any; storedSettings: any }>> {
  if (batch.length === 0) {
    return [];
  }
  const mismatches = await update_embeddings(batch, model, settings, abortController, force, catalogId, anchorId);
  onProcessed(batch.length);
  const { processed, total } = getProgress();
  update_progress_bar(panel, processed, total, settings, 'Computing embeddings');
  return mismatches;
}

/**
 * Apply rate limiting if needed based on page cycle.
 * Common helper to avoid duplication across sweep modes.
 */
async function apply_rate_limit_if_needed(
  hasMore: boolean,
  page: number,
  model: TextEmbeddingModel,
): Promise<void> {
  if (hasMore && (page % model.page_cycle) === 0) {
    console.debug(`Waiting for ${model.wait_period} seconds...`);
    await new Promise(res => setTimeout(res, model.wait_period * 1000));
  }
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
  let selected = '';
  try {
    selected = await joplin.commands.execute('selectedText');
  } catch (error) {
    // Routine error when editor runtime is not available, no need to log
    selected = '';
  }
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
