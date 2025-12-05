import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { EMB_META_KEY, NoteEmbMeta } from '../notes/userDataStore';
import { ensure_catalog_note, load_model_registry, remove_model_from_catalog } from '../notes/catalog';
import { read_model_metadata } from '../notes/catalogMetadataStore';
import { estimate_shard_size } from '../notes/shards';
import { clear_all_corpus_caches } from '../notes/embeddings';
import { clearApiResponse, clearObjectReferences } from '../utils';
import { update_progress_bar } from './panel';
import { JarvisSettings, clear_model_last_sweep_time, clear_model_first_build_completed } from './settings';

const log = getLogger();

const PAGE_SIZE = 100;

interface ModelInventoryItem {
  modelId: string;
  noteCount: number;
  approxBytes: number;
  lastUpdated?: string;
  version?: string;
  isActiveModel: boolean;
}

interface DeletionSummary {
  updatedNotes: number;
  removedMeta: number;
  skippedNotes: number;
  errors: number;
}

/**
 * Escape HTML special characters to prevent user-provided metadata from breaking dialog markup.
 */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Convert a raw byte estimate to a human-readable approximation for UI display.
 */
function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '≈0 KB';
  }
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) {
    return `≈${mb.toFixed(mb >= 10 ? 0 : 1)} MB`;
  }
  const kb = bytes / 1024;
  return `≈${kb.toFixed(kb >= 10 ? 0 : 1)} KB`;
}

/**
 * Render an ISO timestamp in the user's locale, falling back to the original string when parsing fails.
 */
function formatTimestamp(value?: string): string {
  if (!value) {
    return 'unknown';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}

/**
 * Load model inventory from the catalog (fast).
 * This uses the catalog metadata instead of scanning all notes.
 */
async function load_inventory_from_catalog(activeModelId: string): Promise<ModelInventoryItem[]> {
  const catalogId = await ensure_catalog_note();
  const registry = await load_model_registry(catalogId);
  const modelIds = Object.keys(registry).filter(id => registry[id]);

  const items: ModelInventoryItem[] = [];

  for (const modelId of modelIds) {
    const metadata = await read_model_metadata(catalogId, modelId);
    if (!metadata) {
      // Model in registry but no metadata - include with minimal info
      items.push({
        modelId,
        noteCount: 0,
        approxBytes: 0,
        isActiveModel: modelId === activeModelId,
      });
      continue;
    }

    items.push({
      modelId,
      noteCount: metadata.noteCount ?? 0,
      approxBytes: estimate_shard_size(metadata.dim ?? 0, metadata.rowCount ?? 0),
      lastUpdated: metadata.updatedAt,
      version: metadata.version,
      isActiveModel: modelId === activeModelId,
    });
  }

  items.sort((a, b) => a.modelId.localeCompare(b.modelId));
  return items;
}

/**
 * Scan notes to get note IDs for a specific model (needed for deletion).
 * This is only called when actually deleting a model.
 */
async function scan_note_ids_for_model(modelId: string): Promise<string[]> {
  const noteIds: string[] = [];
  let page = 1;
  let hasMore = true;
  let scanned = 0;

  while (hasMore) {
    let response: any = null;
    try {
      response = await joplin.data.get(['notes'], {
        fields: ['id'],
        page,
        limit: PAGE_SIZE,
      });
      const items = response?.items ?? [];
      for (const item of items) {
        const noteId = item?.id as string | undefined;
        if (!noteId) continue;

        scanned += 1;
        if (scanned % 200 === 0) {
          log.debug('Model manager deletion scan progress', { scanned, modelId });
        }

        try {
          const meta = await joplin.data.userDataGet<NoteEmbMeta>(ModelType.Note, noteId, EMB_META_KEY);
          if (meta?.models?.[modelId]) {
            noteIds.push(noteId);
          }
          clearObjectReferences(meta);
        } catch (_) {
          // No userData or error, skip
        }
      }

      hasMore = !!response?.has_more;
      page += 1;
      clearApiResponse(response);
    } catch (err) {
      clearApiResponse(response);
      throw err;
    }
  }

  log.debug('Model manager deletion scan complete', { modelId, noteIds: noteIds.length, scanned });
  return noteIds;
}

/**
 * Delete all shards for a model from a note.
 */
async function delete_shards_for_model(noteId: string, modelId: string, shardCount: number): Promise<void> {
  for (let index = 0; index < shardCount; index += 1) {
    const key = `jarvis/v1/emb/${modelId}/live/${index}` as const;
    try {
      await joplin.data.userDataDelete(ModelType.Note, noteId, key);
    } catch (error) {
      log.warn('Failed to delete shard', { modelId, noteId, index, error });
    }
  }
}

/**
 * Remove embeddings for the specified model from all provided notes and clean up catalog references.
 */
async function delete_model_data(
  modelId: string,
  noteIds: string[],
  panel: string,
  settings: JarvisSettings,
  abortController: AbortController
): Promise<DeletionSummary> {
  const summary: DeletionSummary = {
    updatedNotes: 0,
    removedMeta: 0,
    skippedNotes: 0,
    errors: 0,
  };

  const total = noteIds.length;
  let processed = 0;

  for (const noteId of noteIds) {
    // Check if aborted
    if (abortController.signal.aborted) {
      throw new Error('Model deletion cancelled');
    }
    try {
      const meta = await joplin.data.userDataGet<NoteEmbMeta>(ModelType.Note, noteId, EMB_META_KEY);
      try {
        if (!meta || !meta.models || !meta.models[modelId]) {
          summary.skippedNotes += 1;
          processed += 1;
          continue;
        }

        const modelMeta = meta.models[modelId];
        const shardCount = Math.max(1, modelMeta?.shards ?? 1);
        await delete_shards_for_model(noteId, modelId, shardCount);

        delete meta.models[modelId];

        const remainingModels = Object.keys(meta.models ?? {});
        if (remainingModels.length === 0) {
          await joplin.data.userDataDelete(ModelType.Note, noteId, EMB_META_KEY);
          summary.removedMeta += 1;
        } else {
          await joplin.data.userDataSet(ModelType.Note, noteId, EMB_META_KEY, meta);
          summary.updatedNotes += 1;
        }
      } finally {
        clearObjectReferences(meta);
      }
    } catch (error) {
      summary.errors += 1;
      log.warn('Model manager failed to update note while deleting model', { modelId, noteId, error });
    }

    processed += 1;
    if (processed % 10 === 0 || processed === total) {
      await update_progress_bar(panel, processed, total, settings, `Deleting model "${modelId}"`);
    }
  }

  await remove_model_from_catalog(modelId);
  await clear_model_last_sweep_time(modelId);
  await clear_model_first_build_completed(modelId);

  return summary;
}

/**
 * Remove ALL embeddings from all notes and clear the catalog.
 * This is a complete reset of the embedding database.
 */
async function delete_all_model_data(
  items: ModelInventoryItem[],
  panel: string,
  settings: JarvisSettings,
  abortController: AbortController
): Promise<DeletionSummary> {
  const summary: DeletionSummary = {
    updatedNotes: 0,
    removedMeta: 0,
    skippedNotes: 0,
    errors: 0,
  };

  // Collect all unique note IDs across all models
  await update_progress_bar(panel, 0, 1, settings, 'Scanning notes...');
  const allNoteIds = new Set<string>();
  for (const item of items) {
    // Check if aborted during scanning
    if (abortController.signal.aborted) {
      throw new Error('Model deletion cancelled');
    }
    const noteIds = await scan_note_ids_for_model(item.modelId);
    for (const noteId of noteIds) {
      allNoteIds.add(noteId);
    }
  }

  const total = allNoteIds.size;
  let processed = 0;

  // Delete all userData from each note
  for (const noteId of allNoteIds) {
    // Check if aborted
    if (abortController.signal.aborted) {
      throw new Error('Model deletion cancelled');
    }
    try {
      const meta = await joplin.data.userDataGet<NoteEmbMeta>(ModelType.Note, noteId, EMB_META_KEY);
      try {
        if (!meta || !meta.models) {
          summary.skippedNotes += 1;
          processed += 1;
          continue;
        }

        // Delete shards for all models
        for (const modelId of Object.keys(meta.models)) {
          const modelMeta = meta.models[modelId];
          const shardCount = Math.max(1, modelMeta?.shards ?? 1);
          await delete_shards_for_model(noteId, modelId, shardCount);
        }

        // Delete the metadata entirely
        await joplin.data.userDataDelete(ModelType.Note, noteId, EMB_META_KEY);
        summary.removedMeta += 1;
      } finally {
        clearObjectReferences(meta);
      }
    } catch (error) {
      summary.errors += 1;
      log.warn('Delete all: failed to process note', { noteId, error });
    }

    processed += 1;
    if (processed % 10 === 0 || processed === total) {
      await update_progress_bar(panel, processed, total, settings, 'Deleting all embeddings');
    }
  }

  // Remove all models from catalog and clear per-model settings
  for (const item of items) {
    try {
      await remove_model_from_catalog(item.modelId);
      await clear_model_last_sweep_time(item.modelId);
      await clear_model_first_build_completed(item.modelId);
    } catch (error) {
      log.warn('Delete all: failed to remove model from catalog', { modelId: item.modelId, error });
    }
  }

  // Clear in-memory caches
  clear_all_corpus_caches();

  log.info('Delete all completed', {
    models: items.length,
    notes: allNoteIds.size,
    removedMeta: summary.removedMeta,
    errors: summary.errors
  });

  return summary;
}

/**
 * Build the HTML string for the model management dialog using the computed inventory.
 */
function build_dialog_html(items: ModelInventoryItem[], activeModelId: string, message: string): string {
  if (items.length === 0) {
    return `
      <div id="jarvis-model-manager">
        <h3>Manage Embedding Models</h3>
        <p>No stored embedding models were found.</p>
        <p>Start by indexing your notes via <em>Update Jarvis note DB</em>, then reopen this dialog.</p>
      </div>
    `;
  }

  const header = message
    ? `<div class="jarvis-model-manager__message">${escapeHtml(message)}</div>`
    : '';

  const rows = items.map((item, index) => {
    const activeBadge = item.isActiveModel ? '<span class="jarvis-model-manager__badge">Active</span>' : '';
    const noteSummary = `${item.noteCount}`;
    const checked = index === 0 ? 'checked' : '';
    return `
      <tr>
        <td class="jarvis-model-manager__select">
          <input type="radio" name="modelId" value="${escapeHtml(item.modelId)}" ${checked}>
        </td>
        <td>
          <div class="jarvis-model-manager__model">
            <span class="jarvis-model-manager__model-id">${escapeHtml(item.modelId)}</span>
            ${activeBadge}
          </div>
          <div class="jarvis-model-manager__meta">Last updated: ${escapeHtml(formatTimestamp(item.lastUpdated))}</div>
        </td>
        <td>${escapeHtml(noteSummary)}</td>
        <td>${escapeHtml(formatBytes(item.approxBytes))}</td>
        <td>${escapeHtml(item.version ?? 'unknown')}</td>
      </tr>
    `;
  }).join('');

  return `
    <form name="jarvisModelManager">
      <div id="jarvis-model-manager">
        <h3>Manage Embedding Models</h3>
        <p>Active model: <strong>${escapeHtml(activeModelId || 'unknown')}</strong></p>
        ${header}
        <p class="jarvis-model-manager__hint">Select a model to delete its embeddings from all notes. This cannot be undone, and will require <b>syncing your notes</b>.</p>
        <div class="jarvis-model-manager__table-wrapper">
          <table class="jarvis-model-manager__table">
            <thead>
              <tr>
                <th></th>
                <th>Model</th>
                <th>Notes</th>
                <th>Storage</th>
                <th>Version</th>
              </tr>
            </thead>
            <tbody>
              ${rows}
            </tbody>
          </table>
        </div>
      </div>
    </form>
  `;
}

/**
 * Present the model management dialog workflow, including refresh and deletion actions.
 */
export async function open_model_management_dialog(
  dialogHandle: string,
  panel: string,
  settings: JarvisSettings,
  runtime: any
): Promise<void> {
  const enabled = await joplin.settings.value('notes_db_in_user_data');
  if (!enabled) {
    await joplin.views.dialogs.showMessageBox('Model management requires the experimental per-note userData index to be enabled.');
    return;
  }

  let message = '';

  while (true) {
    const activeModelId = String(await joplin.settings.value('notes_model') ?? '');
    const items = await load_inventory_from_catalog(activeModelId);

    await joplin.views.dialogs.setFitToContent(dialogHandle, false);
    await joplin.views.dialogs.setHtml(dialogHandle, build_dialog_html(items, activeModelId, message));

    const buttons = items.length === 0
      ? [
        { id: 'close', title: 'Close' },
      ]
      : [
        { id: 'delete', title: 'Delete Model' },
        { id: 'deleteAll', title: 'Delete All' },
        { id: 'close', title: 'Close' },
      ];
    await joplin.views.dialogs.setButtons(dialogHandle, buttons);
    await joplin.views.dialogs.setFitToContent(dialogHandle, true);

    const result = await joplin.views.dialogs.open(dialogHandle);
    const action = result?.id ?? 'close';

    if (action === 'close' || items.length === 0) {
      break;
    }

    if (action === 'delete') {
      const formData = result?.formData?.jarvisModelManager;
      const rawSelection = Array.isArray(formData?.modelId)
        ? formData.modelId[0]
        : formData?.modelId;
      const selectedModelId = typeof rawSelection === 'string' ? rawSelection : '';
      if (!selectedModelId) {
        message = 'Select a model before deleting.';
        continue;
      }
      const selected = items.find(item => item.modelId === selectedModelId);
      if (!selected) {
        message = 'Selected model not found. Refresh and try again.';
        continue;
      }
      if (selected.isActiveModel) {
        message = 'Cannot delete the active model. Switch to another model first.';
        continue;
      }

      // Scan for note IDs only when actually deleting
      const noteIds = await scan_note_ids_for_model(selectedModelId);
      const confirm = await joplin.views.dialogs.showMessageBox(
        `Delete model "${selectedModelId}"?\n\nThis removes embeddings from ${noteIds.length} notes. This action cannot be undone.`,
      );
      if (confirm !== 0) {
        message = 'Deletion cancelled.';
        continue;
      }

      // Create abort controller for this deletion operation
      runtime.model_deletion_abort_controller = new AbortController();
      try {
        const summary = await delete_model_data(selectedModelId, noteIds, panel, settings, runtime.model_deletion_abort_controller);
        const parts = [`Removed model "${selectedModelId}".`];
        if (summary.updatedNotes > 0) {
          parts.push(`Updated ${summary.updatedNotes} notes.`);
        }
        if (summary.removedMeta > 0) {
          parts.push(`Cleared metadata on ${summary.removedMeta} notes.`);
        }
        if (summary.skippedNotes > 0) {
          parts.push(`Skipped ${summary.skippedNotes} notes (no metadata).`);
        }
        if (summary.errors > 0) {
          parts.push(`Encountered ${summary.errors} errors (see log).`);
        }
        message = parts.join(' ');
      } catch (error) {
        if (error instanceof Error && error.message.includes('cancelled')) {
          message = 'Deletion cancelled by user.';
        } else {
          throw error;
        }
      } finally {
        runtime.model_deletion_abort_controller = null;
      }
      continue;
    }

    if (action === 'deleteAll') {
      const totalNotes = items.reduce((sum, item) => sum + item.noteCount, 0);
      const totalStorage = items.reduce((sum, item) => sum + item.approxBytes, 0);
      const confirm = await joplin.views.dialogs.showMessageBox(
        `Delete ALL ${items.length} embedding models?\n\n` +
        `This removes all embeddings from ${totalNotes} notes (${formatBytes(totalStorage)}).\n\n` +
        `This action cannot be undone.`,
      );
      if (confirm !== 0) {
        message = 'Deletion cancelled.';
        continue;
      }

      // Create abort controller for this deletion operation
      runtime.model_deletion_abort_controller = new AbortController();
      try {
        const summary = await delete_all_model_data(items, panel, settings, runtime.model_deletion_abort_controller);
        const parts = [`Removed all ${items.length} models.`];
        if (summary.removedMeta > 0) {
          parts.push(`Cleared metadata on ${summary.removedMeta} notes.`);
        }
        if (summary.errors > 0) {
          parts.push(`Encountered ${summary.errors} errors (see log).`);
        }
        message = parts.join(' ');
      } catch (error) {
        if (error instanceof Error && error.message.includes('cancelled')) {
          message = 'Deletion cancelled by user.';
        } else {
          throw error;
        }
      } finally {
        runtime.model_deletion_abort_controller = null;
      }
      continue;
    }

    message = '';
  }
}
