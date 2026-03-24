import joplin from 'api';
import { getLogger } from '../utils/logger';
import type { JarvisSettings } from '../ux/settings';
import { NoteEmbMeta, UserDataEmbStore } from './userDataStore';
import { clearApiResponse } from '../utils';

const log = getLogger();

const DEFAULT_SAMPLE_SIZE = 150;
const MAX_SAMPLE_SIZE = 200;
const MIN_SAMPLE_SIZE = 100;
const NOTE_PAGE_LIMIT = 100;

const userDataStore = new UserDataEmbStore();

export interface ModelCoverageStats {
  totalNotes: number;
  sampledNotes: number;
  notesWithModel: number;
  estimatedNotesWithModel: number;
  coverageRatio: number;
  sampleSizeTarget: number;
}

export const LOW_COVERAGE_THRESHOLD = 0.10;

export type ModelSwitchDecision = 'populate' | 'switch' | 'cancel';

type NotesModelConfig = Pick<JarvisSettings,
  'notes_model' | 'notes_hf_model_id' | 'notes_openai_model_id'>;

/**
 * Resolve the canonical embedding model identifier based on the current Jarvis settings.
 * Returns `null` when the model selection is incomplete to avoid downstream lookups.
 */
export function resolve_embedding_model_id(settings: NotesModelConfig | null | undefined): string | null {
  if (!settings) {
    return null;
  }
  const base = String(settings.notes_model ?? '').trim();
  if (!base) {
    return null;
  }
  if (base === 'Universal Sentence Encoder') {
    return 'Universal Sentence Encoder';
  }
  if (base === 'Hugging Face') {
    const candidate = String(settings.notes_hf_model_id ?? '').trim();
    return candidate || base;
  }
  if (base === 'openai-custom' || base === 'ollama') {
    const candidate = String(settings.notes_openai_model_id ?? '').trim();
    return candidate || base;
  }
  if (base.startsWith('gemini')) {
    const resolved = base.split('-').slice(1).join('-').trim();
    return resolved || base;
  }
  return base;
}

/**
 * Escape HTML special characters so generated dialog markup cannot break rendering or execute scripts.
 */
function escape_html(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Convert a coverage ratio to an integer percentage while clamping to the 0–100 range.
 */
export function coverage_ratio_to_percent(ratio: number): number {
  if (!Number.isFinite(ratio)) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round(ratio * 100)));
}

/**
 * Generate the HTML body for the low-coverage model switch dialog.
 */
export function build_model_switch_dialog_html(stats: ModelCoverageStats, modelId: string): string {
  const percent = coverage_ratio_to_percent(stats.coverageRatio);
  const headline = `Low coverage for ${modelId}`;
  const summary = `Only ~${percent}% of notes (${stats.estimatedNotesWithModel}/${stats.totalNotes}) currently have embeddings. Search results will be limited until embeddings are generated for more notes.`;

  return `
    <form name="jarvisModelSwitch">
      <div id="jarvis-model-switch" class="jarvis-dialog">
        <h3>${escape_html(headline)}</h3>
        <p>${escape_html(summary)}</p>
      </div>
    </form>
  `;
}

/**
 * Prompt the user for how to proceed when switching to a model with low coverage.
 */
export async function prompt_model_switch_decision(
  dialogHandle: string,
  stats: ModelCoverageStats,
  modelId: string,
): Promise<ModelSwitchDecision> {
  await joplin.views.dialogs.setFitToContent(dialogHandle, false);
  await joplin.views.dialogs.setHtml(dialogHandle, build_model_switch_dialog_html(stats, modelId));
  await joplin.views.dialogs.addScript(dialogHandle, 'ux/modelSwitchDialog.css');
  await joplin.views.dialogs.addScript(dialogHandle, 'ux/modelSwitchDialog.js');
  await joplin.views.dialogs.setButtons(dialogHandle, [
    { id: 'populate', title: 'Populate Now' },
    { id: 'switch', title: 'Switch Anyway' },
    { id: 'cancel', title: 'Cancel' },
  ]);
  await joplin.views.dialogs.setFitToContent(dialogHandle, true);
  const result = await joplin.views.dialogs.open(dialogHandle);
  if (result.id === 'populate') {
    return 'populate';
  }
  if (result.id === 'switch') {
    return 'switch';
  }
  return 'cancel';
}

/**
 * Normalize the requested sample size while respecting the 100–200 note guidance.
 */
function sanitize_sample_size(requested: number | undefined): number {
  const target = Number.isFinite(requested) ? Math.round(requested as number) : DEFAULT_SAMPLE_SIZE;
  const clamped = Math.min(Math.max(target, MIN_SAMPLE_SIZE), MAX_SAMPLE_SIZE);
  return clamped;
}

/**
 * Reservoir sample note identifiers across the full library without loading all IDs in memory.
 */
async function reservoir_sample_note_ids(sampleSize: number): Promise<{ total: number; sample: string[] }> {
  const sample: string[] = [];
  let total = 0;
  let page = 0;
  let hasMore = true;

  while (hasMore) {
    page += 1;
    let response: any = null;
    try {
      response = await joplin.data.get(['notes'], { 
        fields: ['id'], 
        limit: NOTE_PAGE_LIMIT, 
        page,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });

      const items: Array<{ id?: string }> = response?.items ?? [];
      hasMore = Boolean(response?.has_more);

      for (const item of items) {
        const id = typeof item?.id === 'string' ? item.id : '';
        if (!id) {
          continue;
        }
        total += 1;
        if (sample.length < sampleSize) {
          sample.push(id);
          continue;
        }
        const index = Math.floor(Math.random() * total);
        if (index < sampleSize) {
          sample[index] = id;
        }
      }

      if (items.length === 0) {
        break;
      }
    } catch (error) {
      log.warn('Failed to fetch note ids during coverage sampling', { page, error });
      break;
    } finally {
      // Clear API response to help GC
      clearApiResponse(response);
    }
  }

  return { total, sample };
}

/**
 * Count how many of the sampled notes contain embeddings for the requested model ID.
 */
async function count_notes_with_model(modelId: string, noteIds: string[]): Promise<number> {
  if (!modelId || noteIds.length === 0) {
    return 0;
  }

  let matches = 0;
  for (const noteId of noteIds) {
    try {
      const meta = await userDataStore.getMeta(noteId);
      if (contains_model(meta, modelId)) {
        matches += 1;
      }
    } catch (error) {
      log.warn('Failed to read metadata while estimating coverage', { noteId, error });
    }
  }
  return matches;
}

/**
 * Determine whether the supplied metadata object contains embeddings for the requested model.
 */
function contains_model(meta: NoteEmbMeta | null, modelId: string): boolean {
  if (!meta || !meta.models) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(meta.models, modelId);
}

/**
 * Estimate coverage for an embedding model by sampling note metadata from userData storage.
 */
export async function estimate_model_coverage(modelId: string, options: { sampleSize?: number } = {}): Promise<ModelCoverageStats> {
  const sampleSizeTarget = sanitize_sample_size(options.sampleSize);

  const { total, sample } = await reservoir_sample_note_ids(sampleSizeTarget);
  if (total === 0) {
    return {
      totalNotes: 0,
      sampledNotes: 0,
      notesWithModel: 0,
      estimatedNotesWithModel: 0,
      coverageRatio: 1,
      sampleSizeTarget,
    };
  }

  const sampledNotes = sample.length;
  const notesWithModel = await count_notes_with_model(modelId, sample);
  const coverageRatio = sampledNotes > 0 ? notesWithModel / sampledNotes : 0;
  const estimatedNotesWithModel = Math.round(coverageRatio * total);

  log.info('Model coverage estimation complete', {
    modelId,
    totalNotes: total,
    sampledNotes,
    notesWithModel,
    coverageRatio,
  });

  return {
    totalNotes: total,
    sampledNotes,
    notesWithModel,
    estimatedNotesWithModel,
    coverageRatio,
    sampleSizeTarget,
  };
}

