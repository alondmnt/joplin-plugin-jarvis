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

// Cache for exclusion tag reverse-lookup (cleared per batch)
let excludedNoteIdsCache: Set<string> | null = null;

/**
 * Get all note IDs with exclusion tags via reverse-lookup.
 * Uses pagination to get all results. Cached per batch.
 */
export async function get_excluded_note_ids_by_tags(): Promise<Set<string>> {
  if (excludedNoteIdsCache) {
    return excludedNoteIdsCache;
  }

  const excludedIds = new Set<string>();
  const exclusionTags = ['jarvis-exclude', 'exclude.from.jarvis'];

  for (const tagTitle of exclusionTags) {
    try {
      // Search for tag by title using query parameter (same pattern as catalog.ts:571)
      const tagSearchResponse = await joplin.data.get(['tags'], {
        query: tagTitle,
        fields: ['id', 'title'],
      });

      // Find exact title match (query may return partial matches)
      const tag = tagSearchResponse.items?.find((t: any) => t.title === tagTitle);
      clearApiResponse(tagSearchResponse);

      if (!tag?.id) {
        // Tag doesn't exist
        continue;
      }

      // Fetch all notes with this tag ID (paginated)
      let page = 1;
      while (true) {
        const notesResponse = await joplin.data.get(['tags', tag.id, 'notes'], {
          fields: ['id'],
          page,
          limit: 100,
        });

        for (const note of notesResponse.items) {
          excludedIds.add(note.id);
        }

        const hasMore = notesResponse.has_more;
        clearApiResponse(notesResponse);

        if (!hasMore) break;
        page++;
      }
    } catch (error) {
      // Tag doesn't exist or fetch failed - continue silently
    }
  }

  excludedNoteIdsCache = excludedIds;
  return excludedIds;
}

/**
 * Clear cached excluded note IDs (call between batches)
 */
export function clear_excluded_note_ids_cache(): void {
  excludedNoteIdsCache = null;
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
  note: { id?: string; parent_id?: string; is_conflict?: boolean; deleted_time?: number },
  noteTags?: string[] | null,
  settings?: JarvisSettings | null,
  options: { checkDeleted?: boolean; checkTags?: boolean; excludedByTag?: Set<string> } = {}
): { excluded: boolean; reason?: string } {
  const { checkDeleted = false, checkTags = true, excludedByTag } = options;

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
  if (checkTags) {
    // Check reverse-lookup first (if provided)
    if (excludedByTag && excludedByTag.has(note.id)) {
      return { excluded: true, reason: 'exclusion tag' };
    }

    // Fall back to checking tags array (for compatibility with other flows)
    if (tags.includes('jarvis-exclude') || tags.includes('exclude.from.jarvis')) {
      const tag = tags.includes('jarvis-exclude') ? 'jarvis-exclude' : 'exclude.from.jarvis';
      return { excluded: true, reason: `${tag} tag` };
    }
  }

  // Check deleted (optional, controlled by checkDeleted flag)
  if (checkDeleted && note.deleted_time && note.deleted_time > 0) {
    return { excluded: true, reason: 'deleted' };
  }

  return { excluded: false };
}
