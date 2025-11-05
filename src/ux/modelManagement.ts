import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { UserDataEmbStore, EMB_META_KEY, NoteEmbMeta } from '../notes/userDataStore';
import { remove_model_from_catalog } from '../notes/catalog';
import { clear_centroid_cache } from '../notes/centroidLoader';

const log = getLogger();

const PAGE_SIZE = 100;
const BYTES_PER_NOTE_ESTIMATE = 300 * 1024; // ≈300KB per note as documented

interface ModelInventoryItem {
  modelId: string;
  noteCount: number;
  approxBytes: number;
  lastUpdated?: string;
  modelVersions: string[];
  embeddingVersions: string[];
  isActiveModel: boolean;
}

interface InventoryResult {
  items: ModelInventoryItem[];
  noteIdsByModel: Map<string, Set<string>>;
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
 * Scan all notes to aggregate per-model inventory statistics and a lookup of note IDs by model.
 */
async function collect_model_inventory(activeModelId: string): Promise<InventoryResult> {
  const store = new UserDataEmbStore(undefined, { cacheSize: 0 });
  const noteIdsByModel = new Map<string, Set<string>>();
  const aggregates = new Map<string, {
    noteCount: number;
    approxBytes: number;
    lastUpdated?: string;
    lastUpdatedEpochMs: number;
    modelVersions: Set<string>;
    embeddingVersions: Set<string>;
    isActiveModel: boolean;
  }>();

  let page = 1;
  let hasMore = true;
  let scanned = 0;

  while (hasMore) {
    const response = await joplin.data.get(['notes'], {
      fields: ['id'],
      page,
      limit: PAGE_SIZE,
      order_by: 'user_updated_time',
      order_dir: 'DESC',
    });
    const items = response?.items ?? [];
    for (const item of items) {
      const noteId = item?.id as string | undefined;
      if (!noteId) {
        continue;
      }
      scanned += 1;
      if (scanned % 200 === 0) {
        log.debug('Model manager inventory scan progress', { scanned });
      }

      let meta: NoteEmbMeta | null = null;
      try {
        meta = await store.getMeta(noteId);
      } catch (error) {
        log.warn('Model manager failed to read metadata', { noteId, error });
      }
      if (!meta || !meta.models) {
        continue;
      }

      for (const [modelId, modelMeta] of Object.entries(meta.models)) {
        if (!modelMeta?.current) {
          continue;
        }

        let notes = noteIdsByModel.get(modelId);
        if (!notes) {
          notes = new Set<string>();
          noteIdsByModel.set(modelId, notes);
        }
        notes.add(noteId);

        let agg = aggregates.get(modelId);
        if (!agg) {
          agg = {
            noteCount: 0,
            approxBytes: 0,
            lastUpdatedEpochMs: 0,
            modelVersions: new Set<string>(),
            embeddingVersions: new Set<string>(),
            isActiveModel: modelId === activeModelId,
          };
          aggregates.set(modelId, agg);
        }
        agg.noteCount += 1;
        agg.approxBytes += BYTES_PER_NOTE_ESTIMATE;
        agg.modelVersions.add(modelMeta.modelVersion ?? 'unknown');
        agg.embeddingVersions.add(String(modelMeta.embeddingVersion ?? 'unknown'));

        const updatedAt = modelMeta.current?.updatedAt;
        if (updatedAt) {
          const epoch = Date.parse(updatedAt);
          if (!Number.isNaN(epoch) && epoch > (agg.lastUpdatedEpochMs ?? 0)) {
            agg.lastUpdatedEpochMs = epoch;
            agg.lastUpdated = updatedAt;
          }
        }
      }
    }

    hasMore = !!response?.has_more;
    page += 1;
  }

  const items: ModelInventoryItem[] = Array.from(aggregates.entries()).map(([modelId, agg]) => ({
    modelId,
    noteCount: agg.noteCount,
    approxBytes: agg.approxBytes,
    lastUpdated: agg.lastUpdated,
    modelVersions: Array.from(agg.modelVersions).sort(),
    embeddingVersions: Array.from(agg.embeddingVersions).sort(),
    isActiveModel: agg.isActiveModel,
  }));

  items.sort((a, b) => a.modelId.localeCompare(b.modelId));

  return { items, noteIdsByModel };
}

/**
 * Remove embeddings for the specified model from all provided notes and clean up catalog references.
 */
async function delete_model_data(modelId: string, noteIds: string[]): Promise<DeletionSummary> {
  const summary: DeletionSummary = {
    updatedNotes: 0,
    removedMeta: 0,
    skippedNotes: 0,
    errors: 0,
  };

  for (const noteId of noteIds) {
    try {
      const meta = await joplin.data.userDataGet<NoteEmbMeta>(ModelType.Note, noteId, EMB_META_KEY);
      if (!meta || !meta.models || !meta.models[modelId]) {
        summary.skippedNotes += 1;
        continue;
      }

      const modelMeta = meta.models[modelId];
      const shardCount = Math.max(1, modelMeta?.current?.shards ?? 1);
      for (let index = 0; index < shardCount; index += 1) {
        const key = `jarvis/v1/emb/${modelId}/live/${index}` as const;
        try {
          await joplin.data.userDataDelete(ModelType.Note, noteId, key);
        } catch (error) {
          log.warn('Model manager failed to delete shard', { modelId, noteId, index, error });
        }
      }

      delete meta.models[modelId];

      const remainingModels = Object.keys(meta.models ?? {});
      if (remainingModels.length === 0) {
        await joplin.data.userDataDelete(ModelType.Note, noteId, EMB_META_KEY);
        summary.removedMeta += 1;
        continue;
      }

      // No need to update activeModelId since it was removed - just save the updated metadata
      await joplin.data.userDataSet(ModelType.Note, noteId, EMB_META_KEY, meta);
      summary.updatedNotes += 1;
    } catch (error) {
      summary.errors += 1;
      log.warn('Model manager failed to update note while deleting model', { modelId, noteId, error });
    }
  }

  await remove_model_from_catalog(modelId);
  clear_centroid_cache(modelId);

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
        <p>No stored embedding models were found in userData.</p>
        <p>Start by indexing your notes via <em>Update Jarvis note DB</em>, then reopen this dialog.</p>
      </div>
    `;
  }

  const header = message
    ? `<div class="jarvis-model-manager__message">${escapeHtml(message)}</div>`
    : '';

  const rows = items.map((item, index) => {
    const versions = item.modelVersions.join(', ');
    const embeddings = item.embeddingVersions.join(', ');
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
        <td>${escapeHtml(versions)}</td>
        <td>${escapeHtml(embeddings)}</td>
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
                <th>Model Version</th>
                <th>Embedding Version</th>
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
export async function open_model_management_dialog(dialogHandle: string): Promise<void> {
  const enabled = await joplin.settings.value('experimental.userDataIndex');
  if (!enabled) {
    await joplin.views.dialogs.showMessageBox('Model management requires the experimental per-note userData index to be enabled.');
    return;
  }

  let message = '';

  while (true) {
    const activeModelId = String(await joplin.settings.value('notes_model') ?? '');
    const { items, noteIdsByModel } = await collect_model_inventory(activeModelId);

    await joplin.views.dialogs.setFitToContent(dialogHandle, false);
    await joplin.views.dialogs.setHtml(dialogHandle, build_dialog_html(items, activeModelId, message));

    const buttons = items.length === 0
      ? [
        { id: 'close', title: 'Close' },
      ]
      : [
        { id: 'delete', title: 'Delete Model' },
        { id: 'refresh', title: 'Refresh' },
        { id: 'close', title: 'Close' },
      ];
    await joplin.views.dialogs.setButtons(dialogHandle, buttons);
    await joplin.views.dialogs.setFitToContent(dialogHandle, true);

    const result = await joplin.views.dialogs.open(dialogHandle);
    const action = result?.id ?? 'close';

    if (action === 'close' || items.length === 0) {
      break;
    }

    if (action === 'refresh') {
      message = 'Inventory refreshed.';
      continue;
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

      const noteIds = Array.from(noteIdsByModel.get(selectedModelId) ?? []);
      const confirm = await joplin.views.dialogs.showMessageBox(
        `Delete model "${selectedModelId}"?\n\nThis removes embeddings from ${noteIds.length} notes and deletes associated centroids. This action cannot be undone.`,
      );
      if (confirm !== 0) {
        message = 'Deletion cancelled.';
        continue;
      }

      const summary = await delete_model_data(selectedModelId, noteIds);
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
      continue;
    }

    message = '';
  }
}

