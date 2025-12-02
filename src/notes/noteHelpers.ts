/**
 * Note-level utilities for retrieving note metadata and content.
 */
import joplin from 'api';
import { clearApiResponse } from '../utils';
import { UserDataEmbStore } from './userDataStore';
import { getLogger } from '../utils/logger';
import type { JarvisSettings } from '../ux/settings';

const log = getLogger();
const ocrMergedFlag = Symbol('ocrTextMerged');

/**
 * Get tags for a note. Returns empty array on error.
 */
export async function get_note_tags(noteId: string): Promise<string[]> {
  let tagsResponse: any = null;
  try {
    tagsResponse = await joplin.data.get(['notes', noteId, 'tags'], { fields: ['title'] });
    const tags = tagsResponse.items.map((t: any) => t.title);
    clearApiResponse(tagsResponse);
    return tags;
  } catch (error) {
    clearApiResponse(tagsResponse);
    return [];
  }
}

/**
 * Get all note IDs that have embeddings in userData.
 * Queries Joplin API for all notes, then filters to those with userData embeddings.
 * Used for candidate selection when experimental userData index is enabled.
 *
 * @param userDataStore - UserData store instance for reading note metadata
 * @param modelId - Optional model ID to count blocks for a specific model
 * @param excludedFolders - Optional set of folder IDs to exclude from results (deprecated, use settings)
 * @param debugMode - Enable debug logging
 * @param settings - Optional Jarvis settings for comprehensive filtering
 * @returns Object with noteIds set and optional totalBlocks count
 */
export async function get_all_note_ids_with_embeddings(
  userDataStore: UserDataEmbStore,
  modelId?: string,
  excludedFolders?: Set<string>,
  debugMode: boolean = false,
  settings?: JarvisSettings,
): Promise<{
  noteIds: Set<string>;
  totalBlocks?: number;
}> {
  const startTime = Date.now();
  const noteIds = new Set<string>();
  let totalBlocks = 0;
  let excludedCount = 0;
  let page = 1;
  let hasMore = true;

  // Support both old excludedFolders param and new settings param for backwards compatibility
  const effectiveSettings = settings || { notes_exclude_folders: excludedFolders || new Set() } as JarvisSettings;

  while (hasMore) {
    let response: any = null;
    try {
      response = await joplin.data.get(['notes'], {
        fields: ['id', 'parent_id', 'deleted_time', 'is_conflict'],
        page,
        limit: 100,
        order_by: 'user_updated_time',
        order_dir: 'DESC',
      });

      for (const note of response.items) {
        // Fetch tags and check all exclusion criteria
        // Cache build is not a hot path, so prefer correctness over micro-optimization
        const tags = await get_note_tags(note.id);
        const result = should_exclude_note(
          note,
          tags,
          effectiveSettings,
          { checkDeleted: true, checkTags: true }
        );

        if (result.excluded) {
          excludedCount++;
          continue;
        }

        // Check if this note has userData embeddings
        const meta = await userDataStore.getMeta(note.id);
        if (meta && meta.models && Object.keys(meta.models).length > 0) {
          noteIds.add(note.id);

          // Count blocks for specific model if requested
          if (modelId && meta.models[modelId]) {
            totalBlocks += meta.models[modelId].current.rows ?? 0;
          }
        }
      }

      hasMore = response.has_more;
      page++;
    } catch (error) {
      log.warn('Failed to fetch note IDs for candidate selection', error);
      break;
    } finally {
      // Clear API response to help GC
      clearApiResponse(response);
    }
  }

  const duration = Date.now() - startTime;
  if (debugMode) {
    log.info(`Candidate selection: found ${noteIds.size} notes with embeddings${modelId ? `, ${totalBlocks} blocks for model ${modelId}` : ''}${excludedCount > 0 ? `, excluded ${excludedCount}` : ''} (took ${duration}ms)`);
  }
  return { noteIds, totalBlocks: modelId ? totalBlocks : undefined };
}

/**
 * Append OCR text from note resources to note body.
 * Processes all resources attached to the note and appends their OCR text.
 * Idempotent - marks note as processed to prevent duplicate processing.
 */
export async function append_ocr_text_to_body(note: any): Promise<void> {
  if (!note || typeof note !== 'object' || note[ocrMergedFlag]) {
    return;
  }

  const body = typeof note.body === 'string' ? note.body : '';
  const noteId = typeof note.id === 'string' ? note.id : undefined;
  let ocrText = '';

  if (noteId) {
    const snippets: string[] = [];
    try {
      let page = 0;
      let resourcesPage: any;
      do {
        page += 1;
        resourcesPage = await joplin.data.get(
          ['notes', noteId, 'resources'],
          { fields: ['id', 'title', 'ocr_text'], page }
        );
        const items = resourcesPage?.items ?? [];
        for (const resource of items) {
          const text = typeof resource?.ocr_text === 'string' ? resource.ocr_text.trim() : '';
          if (text) {
            snippets.push(`\n\n## resource: ${resource.title}\n\n${text}`);
          }
        }
        const hasMore = resourcesPage?.has_more;
        // Clear API response before next iteration
        clearApiResponse(resourcesPage);
        if (!hasMore) break;
      } while (true);
    } catch (error) {
      log.debug(`Failed to retrieve OCR text for note ${noteId}:`, error);
    }
    ocrText = snippets.join('\n\n');
  }

  if (ocrText) {
    const separator = body ? (body.endsWith('\n') ? '\n' : '\n\n') : '';
    note.body = body + separator + ocrText;
  }

  note[ocrMergedFlag] = true;
}

/**
 * Check if a note should be excluded based on filtering criteria.
 * Handles undefined/null values gracefully.
 *
 * @param note - Note object with optional fields
 * @param noteTags - Array of tag titles for this note (can be undefined/null, treated as empty array)
 * @param settings - Jarvis settings containing excluded folders (can be undefined/partial)
 * @param options - Filtering options
 * @param options.checkDeleted - Whether to filter deleted notes (default: false)
 * @param options.checkTags - Whether to check exclusion tags (default: true)
 * @returns Object with { excluded: boolean, reason?: string }
 */
export function should_exclude_note(
  note: { parent_id?: string; is_conflict?: boolean; deleted_time?: number },
  noteTags?: string[] | null,
  settings?: JarvisSettings | null,
  options: { checkDeleted?: boolean; checkTags?: boolean } = {}
): { excluded: boolean; reason?: string } {
  const { checkDeleted = false, checkTags = true } = options;

  // Normalize inputs to handle undefined/null gracefully
  const tags = noteTags || [];
  const excludedFolders = settings?.notes_exclude_folders;

  // Check conflict notes (always checked)
  if (note.is_conflict) {
    return { excluded: true, reason: 'conflict note' };
  }

  // Check excluded folders (always checked, if parent_id and excluded folders are available)
  if (note.parent_id && excludedFolders?.has(note.parent_id)) {
    return { excluded: true, reason: 'excluded folder' };
  }

  // Check exclusion tags (optional, controlled by checkTags flag)
  if (checkTags && (tags.includes('jarvis-exclude') || tags.includes('exclude.from.jarvis'))) {
    const tag = tags.includes('jarvis-exclude') ? 'jarvis-exclude' : 'exclude.from.jarvis';
    return { excluded: true, reason: `${tag} tag` };
  }

  // Check deleted (optional, controlled by checkDeleted flag)
  if (checkDeleted && note.deleted_time && note.deleted_time > 0) {
    return { excluded: true, reason: 'deleted' };
  }

  return { excluded: false };
}
