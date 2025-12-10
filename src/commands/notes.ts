import joplin from 'api';
import { find_nearest_notes, update_embeddings, corpusCaches } from '../notes/embeddings';
import { ensure_catalog_note, register_model, get_catalog_note_id } from '../notes/catalog';
import { update_panel, update_progress_bar } from '../ux/panel';
import { get_settings, mark_model_first_build_completed, get_model_last_sweep_time, set_model_last_sweep_time, set_model_last_full_sweep_time } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { ModelError, clearApiResponse, clearObjectReferences } from '../utils';
import { EmbeddingSettings } from '../notes/userDataStore';
import type { JarvisSettings } from '../ux/settings';
import { getModelStats } from '../notes/modelStats';
import { checkCapacityWarning, SimpleCorpusCache } from '../notes/embeddingCache';
import { read_model_metadata } from '../notes/catalogMetadataStore';
import { should_exclude_note } from '../notes/noteHelpers';

/**
 * Safety margin for incremental sweeps to catch notes that sync late.
 *
 * When an incremental sweep runs, it queries notes modified after:
 *   lastSweepTime - SYNC_SAFETY_MARGIN_MS
 *
 * This creates temporal overlap to handle the race condition where:
 * 1. Sweep runs at time T, sets lastSweepTime=T
 * 2. Sync completes during/after sweep, bringing notes modified before T
 * 3. Next sweep would miss these notes without the margin
 *
 * The 12-hour margin is a best-effort heuristic that handles typical usage:
 * - Daily device switching
 * - Overnight syncs
 * - Notes modified within ~12h that sync late
 *
 * Limitations (by design):
 * - Does NOT catch notes modified >12h ago that sync late
 * - Does NOT catch prolonged offline scenarios (device offline for days)
 * - Only the 12-hour full sweep guarantees eventual consistency
 *
 * The 12-hour value matches FULL_SWEEP_INTERVAL_MS for consistency.
 */
const SYNC_SAFETY_MARGIN_MS = 12 * 60 * 60 * 1000; // 12 hours

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
 *   - Query notes ordered by user_updated_time DESC
 *   - Stop when reaching notes older than (lastSweepTime - SYNC_SAFETY_MARGIN_MS)
 *   - The 12-hour safety margin creates temporal overlap to catch late-arriving syncs
 *   - Typical cost: ~50-100ms for 10-50 updated notes
 *   - Used for: periodic background sweeps to catch sync changes
 *   - Requires: force=false (thoroughness incompatible with early termination)
 *   - Limitation: Only catches notes modified within ~12h of last sweep
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


  // Ensure catalog once before processing notes to prevent
  // concurrent per-note creation races when the experimental userData index is enabled.
  let catalogId: string | undefined;
  if (settings.notes_db_in_user_data && model?.id) {
    try {
      catalogId = await ensure_catalog_note();
      await register_model(catalogId, model.id);
    } catch (error) {
      const msg = String((error as any)?.message ?? error);
      if (!/SQLITE_CONSTRAINT|UNIQUE constraint failed/i.test(msg)) {
        console.debug('Jarvis: deferred ensure catalog (update_note_db)', msg);
      }
    }
  }

  const noteFields = ['id', 'title', 'body', 'is_conflict', 'parent_id', 'deleted_time', 'markup_language'];
  let total_notes = 0;
  let processed_notes = 0;
  let actually_processed_notes = 0;  // Track notes that weren't filtered/skipped
  const allSettingsMismatches: Array<{ noteId: string; currentSettings: any; storedSettings: any }> = [];

  const isFullSweep = !noteIds || noteIds.length === 0;

  // Initialize cache for incremental building during full sweeps
  // Incremental sweeps rely on incremental cache updates to handle synced notes
  if (settings.notes_db_in_user_data && isFullSweep && !incrementalSweep) {
    let cache = corpusCaches.get(model.id);
    if (!cache) {
      cache = new SimpleCorpusCache();
      corpusCaches.set(model.id, cache);
      console.debug('Jarvis: initialized empty cache for incremental build');
    }
    // Don't invalidate - let incremental updates build it during sweep
  }

  // Track total embedding rows created during sweep (for userData model metadata)
  // This avoids loading all embeddings into memory to count them
  let totalEmbeddingRows = 0;
  let embeddingDim = 0;  // Track dimension from first batch (needed when model.embeddings is empty)

  if (noteIds && noteIds.length > 0) {
    // Mode 1: Specific note IDs (note saves, sync notifications)
    const uniqueIds = Array.from(new Set(noteIds));  // dedupe in case of repeated change events
    total_notes = uniqueIds.length;
    // Skip panel updates for small batch updates (< 5 notes) as they complete quickly
    const shouldUpdatePanel = total_notes >= 5;
    if (shouldUpdatePanel) {
      update_progress_bar(panel, 0, total_notes, settings, 'Computing embeddings');
    }

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
        const result = await process_batch_and_update_progress(
          chunk, model, settings, abortController, force, catalogId, panel,
          () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
          (count) => { processed_notes += count; },
          shouldUpdatePanel,
        );
        allSettingsMismatches.push(...result.settingsMismatches);
        totalEmbeddingRows += result.totalRows;
        if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
        actually_processed_notes += result.actuallyProcessed;

        // Clear note bodies after chunk processing
        clearObjectReferences(chunk);
      }
    }

    // Process remaining notes
    if (!abortController.signal.aborted) {
      const result = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, panel,
        () => ({ processed: Math.min(processed_notes, total_notes), total: total_notes }),
        (count) => { processed_notes += count; },
        shouldUpdatePanel,
      );
      allSettingsMismatches.push(...result.settingsMismatches);
      totalEmbeddingRows += result.totalRows;
      if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
      actually_processed_notes += result.actuallyProcessed;

      // Clear remaining batch after processing
      clearObjectReferences(batch);
    }
  } else if (isFullSweep && incrementalSweep && !force) {
    // Mode 2: Timestamp-based incremental sweep (optimized for periodic background updates)
    // Uses user_updated_time (user content changes) instead of updated_time (system changes including userData writes)
    // This prevents reprocessing notes where only embeddings were written, while still catching synced content changes
    // Skip panel updates for incremental sweeps as they are typically quick
    const shouldUpdatePanel = false;
    let page = 1;
    let hasMore = true;
    let reachedOldNotes = false;

    // Apply safety margin to catch notes that synced late (see SYNC_SAFETY_MARGIN_MS docs)
    const cutoffTime = lastSweepTime - SYNC_SAFETY_MARGIN_MS;
    if (settings.notes_debug_mode) {
      console.debug(`Jarvis: incremental sweep cutoff time: ${new Date(cutoffTime).toISOString()} (lastSweepTime - ${SYNC_SAFETY_MARGIN_MS / (60 * 60 * 1000)}h margin)`);
    }

    while (hasMore && !reachedOldNotes && !abortController.signal.aborted) {
      let notes: any = null;
      try {
        notes = await joplin.data.get(['notes'], {
          fields: [...noteFields, 'user_updated_time'],
          page,
          limit: model.page_size,
          order_by: 'user_updated_time',
          order_dir: 'DESC',
        });

        const batch: any[] = [];
        for (const note of notes.items) {
          // Stop when we reach notes older than cutoff (lastSweepTime - safety margin)
          if (note.user_updated_time <= cutoffTime) {
            reachedOldNotes = true;
            if (settings.notes_debug_mode) {
              console.debug(`Reached old notes at page ${page}, stopping early (note timestamp: ${new Date(note.user_updated_time).toISOString()})`);
            }
            break;
          }

          batch.push(note);
        }

      const result = await process_batch_and_update_progress(
        batch, model, settings, abortController, force, catalogId, panel,
        () => ({ processed: processed_notes, total: total_notes }),
        (count) => { processed_notes += count; total_notes += count; },  // Adjust estimate as we go
        shouldUpdatePanel,
      );
      allSettingsMismatches.push(...result.settingsMismatches);
      totalEmbeddingRows += result.totalRows;
      if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
      actually_processed_notes += result.actuallyProcessed;

      // Clear note bodies after batch processing (can be very large)
      // clearObjectReferences on array will clear all elements
      clearObjectReferences(batch);

      hasMore = notes.has_more && !reachedOldNotes;
      page++;

      // Clear API response to help GC (keep notes.items as they're in batch)
      clearApiResponse(notes);

      await apply_rate_limit_if_needed(hasMore, page, model, abortController);
      } finally {
        // Clear API response to help GC
        clearApiResponse(notes);
      }
    }

    console.info(`Jarvis: incremental sweep completed - ${processed_notes} notes processed (${reachedOldNotes ? 'stopped early' : 'reached end of notes'})`);
  } else {
    // Mode 3: Full sweep (thorough scan of all notes)
    // Used for: manual rebuild, force=true validation, first-time builds
    // Always show panel updates for full sweeps as they take longer
    const shouldUpdatePanel = true;
    let notes: any;
    let page = 0;

    // Use last known note count as initial estimate (avoids double-counting loop)
    // Try in-memory stats first, fall back to catalog metadata (persists across restarts)
    const stats = getModelStats(model.id);
    let estimatedTotal = stats?.noteCount ?? 0;
    if (estimatedTotal === 0 && catalogId) {
      const catalogMeta = await read_model_metadata(catalogId, model.id);
      estimatedTotal = catalogMeta?.noteCount ?? 0;
    }

    // Iterate over all notes
    do {
      // Check abort signal before processing next page
      if (abortController.signal.aborted) {
        break;
      }

      page += 1;
      notes = await joplin.data.get(['notes'], {
        fields: noteFields,
        page: page,
        limit: model.page_size,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });
      if (notes.items) {
        // Track only notes with embeddings (not all notes fetched)
        // Use max(estimate, processed) so denominator grows if we exceed the estimate
        const result = await process_batch_and_update_progress(
          notes.items, model, settings, abortController, force, catalogId, panel,
          () => ({
            processed: processed_notes,
            total: Math.max(estimatedTotal, processed_notes)  // Both represent notes with embeddings
          }),
          (count) => {
            processed_notes += count;  // Only increment successfully processed
          },
          shouldUpdatePanel,
        );
        allSettingsMismatches.push(...result.settingsMismatches);
        totalEmbeddingRows += result.totalRows;
        if (embeddingDim === 0 && result.dim > 0) embeddingDim = result.dim;
        actually_processed_notes += result.actuallyProcessed;

        // Clear note bodies after batch processing
        clearObjectReferences(notes.items);
      }

      const hasMoreNotes = notes.has_more;
      // Clear API response before next iteration
      clearApiResponse(notes);

      await apply_rate_limit_if_needed(hasMoreNotes, page, model, abortController);
      
      if (!hasMoreNotes) break;
    } while (true);
    
    console.info(`Jarvis: full sweep completed - ${processed_notes} notes with embeddings processed`);
  }
  
  // Update last sweep timestamp after successful completion of any full sweep
  // (both incremental and thorough scans, but not specific note IDs)
  if (!abortController.signal.aborted && isFullSweep) {
    const timestamp = Date.now();
    await set_model_last_sweep_time(model.id, timestamp);
    console.debug(`Jarvis: updated last sweep time to ${new Date(timestamp).toISOString()}`);

    // Track full sweep timestamp separately for sync staleness detection
    if (!incrementalSweep) {
      await set_model_last_full_sweep_time(model.id, timestamp);
      console.debug(`Jarvis: updated last full sweep time to ${new Date(timestamp).toISOString()}`);
    }
  }

  // Post-sweep metadata update removed (use cache stats instead)
  // Cache stats are updated when cache is built/used during search (see embeddingCache.ts:183)
  // Metadata not critical-path, can be updated lazily with 15% threshold check
  // This saves 10-30s of post-sweep processing time

  // Mark first build as complete after main sweep finishes
  // Only after a FULL sweep (not incremental), which ensures migration/backfill actually happened
  if (
    !abortController.signal.aborted
    && isFullSweep
    && !incrementalSweep
    && settings.notes_db_in_user_data
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
    // Clear SQLite embeddings from memory (migration complete, userData is now authoritative)
    model.embeddings = [];

    await mark_model_first_build_completed(model.id);
    console.info('Jarvis: first userData build completed, disabled legacy SQLite access', { modelId: model.id });
  }

  // Show settings mismatch dialog if mismatches were found during sweep (only for force=false sweeps)
  if (!force && allSettingsMismatches.length > 0 && settings.notes_db_in_user_data) {
    // Deduplicate by noteId
    const uniqueMismatches = Array.from(
      new Map(allSettingsMismatches.map(m => [m.noteId, m])).values()
    );
    
    // Format human-readable summary of settings changes (using first mismatch as representative)
    const settingsDiffs = uniqueMismatches.length > 0
      ? formatSettingsDiff(uniqueMismatches[0].currentSettings, uniqueMismatches[0].storedSettings)
      : [];
    
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
 * Filter out notes that should be excluded from processing.
 * This happens BEFORE the main update pipeline to avoid unnecessary work.
 * 
 * Filters applied:
 * - Deleted notes (deleted_time > 0)
 * - Notes in excluded folders (settings.notes_exclude_folders)
 * - Notes with jarvis-exclude or exclude.from.jarvis tags
 * 
 * Note: Catalog note is automatically excluded via jarvis-exclude tag
 * 
 * @returns Filtered batch with excluded notes removed
 */
async function filter_excluded_notes(
  batch: any[],
  settings: JarvisSettings,
  catalogId: string | undefined,
): Promise<any[]> {
  if (batch.length === 0) {
    return [];
  }
  
  const filtered: any[] = [];
  let excludedCount = 0;

  for (const note of batch) {
    // Only filter conflict notes - they should never be touched
    // All other excluded notes (deleted, excluded folders/tags) flow through to the safety net
    // in update_embeddings() where userData cleanup happens before expensive embedding calculation
    if (note.is_conflict) {
      excludedCount++;
      continue;
    }

    // Let all other notes through (including deleted, excluded folders/tags)
    // The safety net in update_embeddings() will check and clean them up
    filtered.push(note);
  }
  
  if (excludedCount > 0) {
    console.debug(`Jarvis: early filtering excluded ${excludedCount} notes from batch of ${batch.length}, processing ${filtered.length}`);
  }
  
  return filtered;
}

/**
 * Process a batch of notes and update progress bar.
 * Common helper to avoid duplication across sweep modes.
 * Returns settings mismatches and total embedding rows processed.
 */
async function process_batch_and_update_progress(
  batch: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
  force: boolean,
  catalogId: string | undefined,
  panel: string,
  getProgress: () => { processed: number; total: number },
  onProcessed: (count: number) => void,
  shouldUpdatePanel: boolean,
): Promise<{
  settingsMismatches: Array<{ noteId: string; currentSettings: any; storedSettings: any }>;
  totalRows: number;
  dim: number;
  actuallyProcessed: number;
}> {
  if (batch.length === 0) {
    return { settingsMismatches: [], totalRows: 0, dim: 0, actuallyProcessed: 0 };
  }

  // Filter out excluded notes early to avoid unnecessary processing
  const filteredBatch = await filter_excluded_notes(batch, settings, catalogId);

  // Process only the filtered batch
  const result = await update_embeddings(filteredBatch, model, settings, abortController, force, catalogId);

  // Track actually processed notes (after filtering but before skipping)
  const actuallyProcessed = filteredBatch.length;

  // Report progress based on actual processed count (excludes excluded/deleted notes)
  // This ensures progress bar numerator matches the denominator (notes with embeddings)
  onProcessed(result.processedCount);
  if (shouldUpdatePanel) {
    const { processed, total } = getProgress();
    update_progress_bar(panel, processed, total, settings, 'Computing embeddings');
  }
  return { ...result, actuallyProcessed };
}

/**
 * Apply rate limiting if needed based on page cycle.
 * Common helper to avoid duplication across sweep modes.
 * Polls abort signal during wait for responsive cancellation.
 */
async function apply_rate_limit_if_needed(
  hasMore: boolean,
  page: number,
  model: TextEmbeddingModel,
  abortController: AbortController,
): Promise<void> {
  if (hasMore && (page % model.page_cycle) === 0) {
    console.debug(`Pause checkpoint at page ${page} (${model.wait_period}s)...`);

    // Poll abort signal during wait period for immediate cancellation
    const pollInterval = 50; // Check every 50ms
    const totalWait = model.wait_period * 1000;
    const polls = Math.ceil(totalWait / pollInterval);

    for (let i = 0; i < polls; i++) {
      if (abortController.signal.aborted) {
        console.debug(`Cancel detected during pause at page ${page}`);
        return; // Exit immediately without completing wait
      }
      await new Promise(res => setTimeout(res, pollInterval));
    }
  }
}

/**
 * Format differences between two sets of embedding settings into human-readable strings.
 * Helper to avoid duplication when displaying settings changes to users.
 * 
 * @returns Array of strings describing each setting that differs (e.g. "embedTitle No→Yes")
 */
function formatSettingsDiff(current: EmbeddingSettings, stored: EmbeddingSettings): string[] {
  const diffs: string[] = [];
  const boolFields: Array<keyof EmbeddingSettings> = [
    'embedTitle',
    'embedPath',
    'embedHeading',
    'embedTags',
    'includeCode'
  ];
  
  for (const field of boolFields) {
    if (current[field] !== stored[field]) {
      const storedValue = stored[field] ? 'Yes' : 'No';
      const currentValue = current[field] ? 'Yes' : 'No';
      diffs.push(`${field} ${storedValue}→${currentValue}`);
    }
  }
  
  if (current.maxTokens !== stored.maxTokens) {
    diffs.push(`maxTokens ${stored.maxTokens}→${current.maxTokens}`);
  }
  
  return diffs;
}

export async function find_notes(model: TextEmbeddingModel, panel: string, explicit: boolean = false) {
  if (!(await joplin.views.panels.visible(panel))) {
    return;
  }
  if (model.model === null) { return; }
  const settings = await get_settings();

  const note = await joplin.workspace.selectedNote();
  if (!note) {
    return;
  }

  // Skip search for excluded notes on automatic triggers (respect user's exclusion intent).
  // Explicit triggers (e.g., toggle panel command) bypass this check.
  if (!explicit) {
    const exclusionResult = should_exclude_note(note, null, settings,
      { checkDeleted: true, checkTags: true });
    if (exclusionResult.excluded) {
      await update_panel(panel, [], settings, null,
        "Note excluded. Run 'Find related notes' to search anyway.");
      return;
    }
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
    nearest = await find_nearest_notes(model.embeddings, note.id, note.markup_language, note.title, selected, model, settings, true, panel);
  } catch (error) {
    if (error instanceof ModelError) {
      await joplin.views.dialogs.showMessageBox(`Error: ${error.message}`);
      return;
    }
    throw error;
  }

  // Compute capacity warning from in-memory stats (if available)
  const stats = getModelStats(model.id);
  const profileIsDesktop = settings.notes_device_profile_effective === 'desktop';
  const capacityWarning = stats
    ? checkCapacityWarning(stats.rowCount, stats.dim, profileIsDesktop)
    : null;

  // write results to panel
  await update_panel(panel, nearest, settings, capacityWarning);

  // Clear note body after use
  clearObjectReferences(note);
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
  const storageMethod = settings.notes_db_in_user_data 
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
