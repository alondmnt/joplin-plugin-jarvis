/**
 * Note embedding update orchestration.
 * Handles updating embeddings for notes including:
 * - Content change detection via hashing
 * - Incremental vs full rebuild logic
 * - Storage (SQLite legacy or userData)
 * - Cache invalidation
 * - Batch processing with error handling
 */
import joplin from 'api';
import { ModelError, clearApiResponse, clearObjectReferences } from '../utils';
import { JarvisSettings } from '../ux/settings';
import { TextEmbeddingModel } from '../models/models';
import { getLogger } from '../utils/logger';
import { UserDataEmbStore, EmbeddingSettings } from './userDataStore';
import { prepare_user_data_embeddings } from './userDataIndexer';
import { extract_embedding_settings_for_validation, settings_equal } from './validator';
import { append_ocr_text_to_body, should_exclude_note, get_excluded_note_ids_by_tags, clear_excluded_note_ids_cache } from './noteHelpers';
import type { BlockEmbedding } from './embeddings';
import { calc_note_embeddings, calc_hash, userDataStore, preprocess_note_for_hashing } from './embeddings';
import { delete_note_and_embeddings, insert_note_embeddings } from './db';
import { update_cache_for_note } from './embeddingCache';
import { htmlToText } from '../utils';

const log = getLogger();

export interface UpdateNoteResult {
  embeddings: BlockEmbedding[];
  settingsMismatch?: {
    noteId: string;
    currentSettings: EmbeddingSettings;
    storedSettings: EmbeddingSettings;
  };
  skippedUnchanged?: boolean; // True if note was skipped due to matching hash and settings
}

type EmbeddingErrorAction = 'retry' | 'skip' | 'abort';

export const MAX_EMBEDDING_RETRIES = 2;

function formatNoteLabel(note: { id: string; title?: string }): string {
  return note.title ? `${note.id} (${note.title})` : note.id;
}

export function ensure_model_error(
  rawError: unknown,
  context?: { id?: string; title?: string },
): ModelError {
  const baseMessage = rawError instanceof Error ? rawError.message : String(rawError);
  const noteId = context?.id;
  const label = noteId ? formatNoteLabel({ id: noteId, title: context?.title }) : null;
  const message = (label && noteId)
    ? (baseMessage.includes(noteId) ? baseMessage : `Note ${label}: ${baseMessage}`)
    : baseMessage;

  if (rawError instanceof ModelError) {
    if (rawError.message === message) {
      return rawError;
    }
    const enriched = new ModelError(message);
    (enriched as any).cause = (rawError as any).cause ?? rawError;
    return enriched;
  }

  const modelError = new ModelError(message);
  (modelError as any).cause = rawError;
  return modelError;
}

export async function promptEmbeddingError(
  settings: JarvisSettings,
  error: ModelError,
  options: {
    attempt: number;
    maxAttempts: number;
    allowSkip: boolean;
    skipLabel?: string;
  },
): Promise<EmbeddingErrorAction> {
  if (settings.notes_abort_on_error) {
    await joplin.views.dialogs.showMessageBox(`Error: ${error.message}`);
    return 'abort';
  }

  const { attempt, maxAttempts, allowSkip, skipLabel } = options;

  if (attempt < maxAttempts) {
    const cancelAction = allowSkip ? (skipLabel ?? 'skip this note') : 'cancel this operation.';
    const message = allowSkip
      ? `Error: ${error.message}\nPress OK to retry or Cancel to ${cancelAction}.`
      : `Error: ${error.message}\nPress OK to retry or Cancel to ${cancelAction}`;
    const choice = await joplin.views.dialogs.showMessageBox(message);
    if (choice === 0) {
      return 'retry';
    }
    return allowSkip ? 'skip' : 'abort';
  }

  const message = allowSkip
    ? `Error: ${error.message}\nAlready tried ${attempt + 1} times.\nPress OK to skip this note or Cancel to abort.`
    : `Error: ${error.message}\nAlready tried ${attempt + 1} times.\nPress OK to retry again or Cancel to cancel this operation.`;
  const choice = await joplin.views.dialogs.showMessageBox(message);
  if (allowSkip) {
    return (choice === 0) ? 'skip' : 'abort';
  }
  return (choice === 0) ? 'retry' : 'abort';
}

async function write_user_data_embeddings(
  note: any,
  blocks: BlockEmbedding[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  contentHash: string,
  catalogId?: string,
): Promise<void> {
  const noteId = note?.id;
  if (!noteId) {
    return;
  }
  try {
    const prepared = await prepare_user_data_embeddings({
      noteId,
      contentHash,
      blocks,
      model,
      settings,
      store: userDataStore,
      catalogId,
    });
    if (!prepared) {
      return;
    }
    await userDataStore.put(noteId, model.id, prepared.meta, prepared.shards);

    // Clear large shard data after PUT (shards contain quantized vectors - can be large)
    // clearObjectReferences will set shards array length to 0, releasing all shard references
    clearObjectReferences(prepared);
  } catch (error) {
    log.warn(`Failed to persist userData embeddings for note ${noteId}`, error);
  }
}

async function update_note(note: any,
    model: TextEmbeddingModel, settings: JarvisSettings,
    abortSignal: AbortSignal, force: boolean = false, catalogId?: string, excludedByTag?: Set<string>): Promise<UpdateNoteResult> {
  if (abortSignal.aborted) {
    throw new ModelError("Operation cancelled");
  }

  // Check exclusion using reverse-lookup (no per-note tag fetch!)
  // This safety net handles exclusion before expensive embedding calculation
  const exclusionResult = should_exclude_note(
    note,
    null,  // No tag array (using reverse-lookup instead)
    settings,
    { checkDeleted: true, checkTags: true, excludedByTag }
  );

  if (exclusionResult.excluded) {
    // Log why this note is excluded (helps identify repeated processing or filter bypass)
    if (settings.notes_debug_mode) {
      log.debug(`Late exclusion (safety check): note ${note.id} - reason: ${exclusionResult.reason}`);
    }
    delete_note_and_embeddings(model.db, note.id);

    // Delete all userData embeddings (for all models)
    if (settings.notes_db_in_user_data) {
      try {
        await userDataStore.gcOld(note.id, '', '');
      } catch (error) {
        log.warn(`Failed to delete userData for excluded note ${note.id}`, error);
      }

      // Incrementally update cache (remove blocks for deleted note)
      await update_cache_for_note(userDataStore, model.id, note.id, '', settings.notes_debug_mode, true);
    }

    return { embeddings: [] };
  }

  // Preprocess note body (HTML conversion + OCR appending)
  note.body = await preprocess_note_for_hashing(note);

  const hash = calc_hash(note.body);
  const old_embd = model.embeddings.filter((embd: BlockEmbedding) => embd.id === note.id);

  // Fetch userData meta once and cache it for this update (avoid multiple reads)
  let userDataMeta: Awaited<ReturnType<typeof userDataStore.getMeta>> | null = null;
  if (settings.notes_db_in_user_data) {
    try {
      userDataMeta = await userDataStore.getMeta(note.id);
    } catch (error) {
      log.debug(`Failed to fetch userData for note ${note.id}`, error);
    }
  }

  // Check abort after potentially slow userData fetch (improves cancel responsiveness)
  if (abortSignal.aborted) {
    throw new ModelError("Operation cancelled");
  }

  // Check if content unchanged (check both SQLite and userData)
  let hashMatch = (old_embd.length > 0) && (old_embd[0].hash === hash);
  let userDataHashMatch = false;

  // When userData is enabled and SQLite is empty, check userData for existing hash
  if (!hashMatch && userDataMeta && old_embd.length === 0) {
    const modelMeta = userDataMeta?.models?.[model.id];
    if (modelMeta && modelMeta.contentHash === hash) {
      hashMatch = true;
      userDataHashMatch = true;
    }
  }

  if (hashMatch) {
    // Content unchanged - decide whether to skip based on force parameter

    if (!force) {
      // force=false (note save/sweep): Skip if content unchanged, but validate settings
      // This is for incremental updates when user saves a note or background sweeps

      // Backfill from SQLite when userData is completely missing (migration path)
      if (!userDataMeta && old_embd.length > 0 && settings.notes_db_in_user_data) {
        if (settings.notes_debug_mode) {
          log.debug(`Note ${note.id} needs backfill from SQLite - no userData exists`);
        }
        await write_user_data_embeddings(note, old_embd, model, settings, hash, catalogId);

        // Update cache after backfill
        await update_cache_for_note(userDataStore, model.id, note.id, hash, settings.notes_debug_mode);

        return { embeddings: old_embd, skippedUnchanged: true };
      }

      if (userDataMeta) {
        const modelMeta = userDataMeta?.models?.[model.id];
        let needsBackfill = !userDataMeta
          || !modelMeta
          || modelMeta.contentHash !== hash;
        let needsCompaction = false;
        let shardMissing = false;
        if (!needsBackfill && userDataMeta && modelMeta && modelMeta.shards > 0) {
          let first: any = null;
          try {
            first = await userDataStore.getShard(note.id, model.id, 0, userDataMeta);  // Pass meta to avoid redundant getMeta()
            if (!first) {
              // Metadata exists but shard is missing/corrupt - needs backfill
              shardMissing = true;
              if (settings.notes_debug_mode) {
                log.debug(`Note ${note.id} has metadata but missing/invalid shard - will backfill`);
              }
            } else {
              const row0 = first?.meta?.[0] as any;
              // Detect legacy rows by presence of duplicated per-row fields or blockId
              needsCompaction = Boolean(row0?.noteId || row0?.noteHash || row0?.blockId);

              // Clear base64 strings from shard after inspection
              delete (first as any).vectorsB64;
              delete (first as any).scalesB64;
            }
          } catch (e) {
            // Shard read failed - treat as missing/corrupt
            shardMissing = true;
            log.debug(`Note ${note.id} shard read failed - will backfill`, e);
          } finally {
            clearObjectReferences(first);
          }
        }
        if (needsBackfill || needsCompaction || shardMissing) {
          if (settings.notes_debug_mode) {
            log.debug(`Note ${note.id} needs backfill/compaction - needsBackfill=${needsBackfill}, needsCompaction=${needsCompaction}, shardMissing=${shardMissing}`);
          }
          await write_user_data_embeddings(note, old_embd, model, settings, hash, catalogId);
        }

        // Validate settings even when content unchanged (catches synced mismatches)
        // Check if this note has embeddings for OUR model, regardless of which model is "active"
        if (userDataMeta && modelMeta) {
          const currentSettings = extract_embedding_settings_for_validation(settings);

          // Check if settings match for our model's embeddings
          if (!settings_equal(currentSettings, modelMeta.settings)) {
            // Settings mismatch - return mismatch info for dialog
            log.info(`Note ${note.id}: settings mismatch detected during sweep for model ${model.id}`);
            return {
              embeddings: old_embd,
              settingsMismatch: {
                noteId: note.id,
                currentSettings,
                storedSettings: modelMeta.settings,
              },
            };
          }
        }

        // Before returning skippedUnchanged, update cache with existing userData
        // This ensures cache includes recently-synced notes even if embeddings are up-to-date
        if (settings.notes_db_in_user_data) {
          await update_cache_for_note(userDataStore, model.id, note.id, hash, settings.notes_debug_mode);
        }

        return { embeddings: old_embd, skippedUnchanged: true }; // Skip - content unchanged, settings match
      }
    }

    // force=true: Check if userData matches current settings/model before skipping
    // This is for manual "Update DB", settings changes, or validation dialog rebuilds
    if (userDataMeta) {
      // Check abort before expensive force=true validation (improves cancel responsiveness)
      if (abortSignal.aborted) {
        throw new ModelError("Operation cancelled");
      }

      if (userDataMeta.models[model.id]) {
        const modelMeta = userDataMeta.models[model.id];
        const currentSettings = extract_embedding_settings_for_validation(settings);

        const hashMatches = modelMeta.contentHash === hash;
        const modelVersionMatches = modelMeta.modelVersion === (model.version ?? 'unknown');
        const embeddingVersionMatches = modelMeta.embeddingVersion === (model.embedding_version ?? 0);
        const settingsMatch = settings_equal(currentSettings, modelMeta.settings);

        if (modelMeta && hashMatches && modelVersionMatches && embeddingVersionMatches && settingsMatch) {
          // Everything appears up-to-date, but check for incomplete shards
          let shardMissing = false;
          if (modelMeta.shards > 0) {
            let first: any = null;
            try {
              first = await userDataStore.getShard(note.id, model.id, 0, userDataMeta);  // Pass meta to avoid redundant getMeta()

              // Check abort after potentially slow shard fetch (improves cancel responsiveness)
              if (abortSignal.aborted) {
                clearObjectReferences(first);
                throw new ModelError("Operation cancelled");
              }

              if (!first) {
                // Metadata exists but shard is missing/corrupt - needs rebuild
                shardMissing = true;
                log.info(`Note ${note.id} has incomplete shard (metadata exists but shard data missing) - will rebuild`);
              }
            } catch (e) {
              // Check if this is a cancellation vs actual error
              if (e instanceof ModelError && e.message === "Operation cancelled") {
                throw e;
              }
              // Shard read failed - treat as missing/corrupt
              shardMissing = true;
              log.info(`Note ${note.id} shard read failed - will rebuild`, e);
            } finally {
              clearObjectReferences(first);
            }
          }

          if (!shardMissing) {
            // Everything up-to-date: content, settings, model, and shards all valid - skip
            // This is the expected behavior: force=true means "recheck everything", not "rebuild everything"
            // Note: old_embd may be empty when userData is enabled (embeddings stored in userData, not SQLite)

            // Update cache with existing userData
            if (settings.notes_db_in_user_data) {
              await update_cache_for_note(userDataStore, model.id, note.id, hash, settings.notes_debug_mode);
            }

            return { embeddings: old_embd, skippedUnchanged: true };
          }

          // Shard incomplete/missing - fall through to rebuild
          log.info(`Note ${note.id} needs rebuild due to incomplete shard data`);
        }
        // userData exists but outdated (settings or model changed) - rebuild needed
        log.info(`Rebuilding note ${note.id} - userData outdated (hash=${hashMatches}, model=${modelVersionMatches}, embedding=${embeddingVersionMatches}, settings=${settingsMatch})`);
      } else {
        // userData missing or wrong model - rebuild needed
        log.info(`Rebuilding note ${note.id} - no userData for model ${model.id}`);
      }
    } else {
      // userData missing - rebuild needed
      log.info(`Rebuilding note ${note.id} - no userData found`);
    }

    if (!settings.notes_db_in_user_data) {
      // notes_db_in_user_data disabled - skip since content unchanged
      return { embeddings: old_embd, skippedUnchanged: true };
    }

    // Backfill from SQLite when userData is missing but SQLite has valid embeddings
    // This enables migration without re-embedding (saves API quota)
    if (old_embd.length > 0) {
      if (settings.notes_debug_mode) {
        log.debug(`Note ${note.id} - backfilling from SQLite to userData (migration)`);
      }
      await write_user_data_embeddings(note, old_embd, model, settings, hash, catalogId);

      // Update cache after backfill
      await update_cache_for_note(userDataStore, model.id, note.id, hash, settings.notes_debug_mode);

      return { embeddings: old_embd, skippedUnchanged: true };
    }
  }

  // Rebuild needed: content changed OR (force=true AND userData outdated/missing AND no SQLite to backfill)

  // Fetch tags ONLY if needed for embedding (after hash check determined re-embedding is needed)
  let note_tags: string[] = [];
  if (settings.notes_embed_tags) {
    let tagsResponse: any = null;
    try {
      tagsResponse = await joplin.data.get(['notes', note.id, 'tags'], { fields: ['title'] });
      note_tags = tagsResponse.items.map((t: any) => t.title);
      clearApiResponse(tagsResponse);
    } catch (error) {
      clearApiResponse(tagsResponse);
      note_tags = [];
    }
  }

  try {
    const new_embd = await calc_note_embeddings(note, note_tags, model, settings, abortSignal, 'doc');

    // Write embeddings to appropriate storage
    if (settings.notes_db_in_user_data) {
      // userData mode: Write to userData
      await write_user_data_embeddings(note, new_embd, model, settings, hash, catalogId);

      // Incrementally update cache (replace blocks for updated note)
      await update_cache_for_note(userDataStore, model.id, note.id, hash, settings.notes_debug_mode, true);
    } else {
      // Legacy mode: SQLite is primary storage, clean up any old userData
      await insert_note_embeddings(model.db, new_embd, model);
      // Clean up ALL userData (metadata + shards for all models) when feature is disabled
      // This prevents stale userData from accumulating and causing confusion
      if (note?.id) {
        userDataStore.gcOld(note.id, '', '').catch(error => {
          log.debug(`Failed to clean userData for note ${note.id}`, error);
        });
      }
    }

    return { embeddings: new_embd };
  } catch (error) {
    throw ensure_model_error(error, note);
  }
}

// in-place function
/**
 * Update embeddings for multiple notes.
 *
 * @param force - Controls rebuild behavior for all notes:
 *   - false: Skip notes where content unchanged, but validate settings (returns mismatches)
 *   - true: Skip only if content unchanged AND settings match AND model matches
 *
 * Processes notes in batches, handling errors per-note with retry/skip/abort prompts.
 * Updates in-memory model.embeddings array after successful batch completion.
 *
 * @returns Object containing settings mismatches and total embedding rows processed
 */
export async function update_embeddings(
  notes: any[],
  model: TextEmbeddingModel,
  settings: JarvisSettings,
  abortController: AbortController,
  force: boolean = false,
  catalogId?: string,
): Promise<{
  settingsMismatches: Array<{ noteId: string; currentSettings: EmbeddingSettings; storedSettings: EmbeddingSettings }>;
  totalRows: number;
  dim: number;
  processedCount: number;
}> {
  // Fetch excluded note IDs once for entire batch (with pagination)
  const excludedByTag = await get_excluded_note_ids_by_tags();

  const successfulNotes: Array<{ note: any; embeddings: BlockEmbedding[] }> = [];
  const skippedNotes: string[] = [];
  const skippedUnchangedNotes: string[] = []; // Notes skipped due to matching hash and settings
  const settingsMismatches: Array<{ noteId: string; currentSettings: EmbeddingSettings; storedSettings: EmbeddingSettings }> = [];
  let dialogQueue: Promise<unknown> = Promise.resolve();
  let fatalError: ModelError | null = null;
  const runSerialized = async <T>(fn: () => Promise<T>): Promise<T> => {
    const next = dialogQueue.then(fn);
    dialogQueue = next.catch(() => undefined);
    return next;
  };

  const notePromises = notes.map(async note => {
    let attempt = 0;
    while (!abortController.signal.aborted) {
      try {
        const result = await update_note(note, model, settings, abortController.signal, force, catalogId, excludedByTag);
        successfulNotes.push({ note, embeddings: result.embeddings });

        // Track notes that were skipped due to matching hash and settings
        if (result.skippedUnchanged) {
          skippedUnchangedNotes.push(note.id);
        }

        // Collect settings mismatches (only during force=false sweeps)
        if (result.settingsMismatch) {
          settingsMismatches.push(result.settingsMismatch);
        }
        return;
      } catch (rawError) {
        const error = ensure_model_error(rawError, note);

        if (fatalError) {
          throw fatalError;
        }

        const action = await runSerialized(() =>
          promptEmbeddingError(settings, error, {
            attempt,
            maxAttempts: MAX_EMBEDDING_RETRIES,
            allowSkip: true,
            skipLabel: 'skip this note',
          })
        );

        if (action === 'abort') {
          fatalError = fatalError ?? error;
          abortController.abort();
          throw fatalError;
        }

        if (action === 'retry') {
          attempt += 1;
          continue;
        }

        if (action === 'skip') {
          log.warn(`Skipping note ${note.id}: ${error.message}`, (error as any).cause ?? error);
          skippedNotes.push(note.id);
          return;
        }
      }
    }

    throw fatalError ?? new ModelError('Model embedding operation cancelled');
  });

  await Promise.all(notePromises);

  if (notes.length > 0) {
    const successCount = successfulNotes.length;
    const skipCount = skippedNotes.length;
    const unchangedCount = skippedUnchangedNotes.length;
    const failCount = notes.length - successCount - skipCount;

    // Only log batch completion if there are issues or in debug mode
    if (settings.notes_debug_mode || failCount > 0 || skipCount > 0) {
      log.info(`Batch complete: ${successCount} successful (${unchangedCount} unchanged), ${skipCount} skipped, ${failCount} failed of ${notes.length} total`);
    }

    if (skipCount > 0) {
      log.warn(`Skipped note IDs: ${skippedNotes.slice(0, 10).join(', ')}${skipCount > 10 ? ` ... and ${skipCount - 10} more` : ''}`);
    }
  }

  // Count notes with embeddings (new/updated) OR unchanged notes (validated with hash match)
  // This excludes only excluded/deleted notes that return empty embeddings without skippedUnchanged flag
  const processedCount = successfulNotes.filter(item =>
    item.embeddings.length > 0 || skippedUnchangedNotes.includes(item.note.id)
  ).length;

  if (successfulNotes.length === 0) {
    return { settingsMismatches, totalRows: 0, dim: 0, processedCount: 0 };
  }

  // Only populate model.embeddings when userData index is disabled (legacy mode)
  // When userData is enabled, search reads directly from userData (memory efficient)
  if (!settings.notes_db_in_user_data) {
    const mergedEmbeddings = successfulNotes.flatMap(result => result.embeddings);
    remove_note_embeddings(
      model.embeddings,
      successfulNotes.map(result => result.note.id),
    );
    model.embeddings.push(...mergedEmbeddings);
    const dim = mergedEmbeddings[0]?.embedding?.length ?? 0;

    // Help GC by clearing the batch data (note bodies can be large)
    for (const item of successfulNotes) {
      if (item.note) {
        clearObjectReferences(item.note);
      }
    }
    clearObjectReferences(successfulNotes);

    return { settingsMismatches, totalRows: mergedEmbeddings.length, dim, processedCount };
  }

  // Count total embedding rows without creating temporary array (memory efficient)
  const totalRows = successfulNotes.reduce((sum, result) => sum + result.embeddings.length, 0);

  // Get dimension from first embedding (needed for model metadata when model.embeddings is empty)
  const dim = successfulNotes[0]?.embeddings[0]?.embedding?.length ?? 0;

  // Help GC by clearing batch data (note bodies can be large)
  for (const item of successfulNotes) {
    if (item.note) {
      clearObjectReferences(item.note);
    }
  }
  clearObjectReferences(successfulNotes);

  // Clear excluded note IDs cache after batch
  clear_excluded_note_ids_cache();

  return { settingsMismatches, totalRows, dim, processedCount };
}

// function to remove all embeddings of the given notes from an array of embeddings in-place
function remove_note_embeddings(embeddings: BlockEmbedding[], note_ids: string[]) {
  let end = embeddings.length;
  const note_ids_set = new Set(note_ids);

  for (let i = 0; i < end; ) {
    if (note_ids_set.has(embeddings[i].id)) {
      [embeddings[i], embeddings[end-1]] = [embeddings[end-1], embeddings[i]]; // swap elements
      end--;
    } else {
      i++;
    }
  }

  embeddings.length = end;
}
