/**
 * Inter-plugin API commands.
 *
 * Registers Joplin commands under the `jarvis.api.*` namespace that return
 * serialisable JSON. Other plugins call them via:
 *
 *   const result = await joplin.commands.execute('jarvis.api.search', { query: 'foo' });
 *
 * All responses follow { ok: true, ... } | { ok: false, error, message }.
 */
import joplin from 'api';
import { getLogger } from '../utils/logger';
import { clearApiResponse, htmlToText } from '../utils';
import { find_nearest_notes, append_ocr_text_to_body } from '../notes/embeddings';
import type { NoteEmbedding } from '../notes/embeddings';
import type { TextEmbeddingModel } from '../models/models';
import type { JarvisSettings } from '../ux/settings';
import { getModelStats } from '../notes/modelStats';
import { corpusCaches } from '../notes/embeddingCache';

const log = getLogger();

const API_VERSION = 1;

/** Narrow interface — avoids coupling to the full PluginRuntime. */
export interface ApiRuntime {
  model_embed: TextEmbeddingModel;
  settings: JarvisSettings;
}

// ── helpers ─────────────────────────────────────────────────────────

function errorResponse(error: string, message: string) {
  return { ok: false, error, message };
}

function isReady(runtime: ApiRuntime): boolean {
  return runtime.model_embed.model !== null;
}

/** Strip Float32Array / q8 from NoteEmbedding[], keep only serialisable metadata + scores. */
function toSerializableResults(notes: NoteEmbedding[]) {
  return notes.map(n => ({
    noteId: n.id,
    noteTitle: n.title,
    similarity: n.similarity,
    blocks: n.embeddings
      .slice()
      .sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0))
      .map(b => ({
        title: b.title,
        line: b.line,
        level: b.level,
        bodyIdx: b.body_idx,
        length: b.length,
        similarity: b.similarity ?? 0,
        text: '',  // filled in by attachBlockText
      })),
  }));
}

/** Fetch note bodies and fill in block text from stored offsets. */
async function attachBlockText(
  results: ReturnType<typeof toSerializableResults>,
): Promise<void> {
  // Collect unique noteIds
  const noteIds = [...new Set(results.map(r => r.noteId))];
  const bodies = new Map<string, string>();

  for (const id of noteIds) {
    let resp: any = null;
    try {
      resp = await joplin.data.get(['notes', id], { fields: ['body', 'markup_language'] });
      let body: string = resp.body ?? '';
      if (resp.markup_language === 2) {
        try { body = await htmlToText(body); } catch { /* keep original */ }
      }
      // Append OCR text to match the body offsets used during indexing
      const noteObj: any = { id, body, markup_language: resp.markup_language };
      await append_ocr_text_to_body(noteObj);
      bodies.set(id, noteObj.body);
      clearApiResponse(resp);
    } catch {
      clearApiResponse(resp);
      // Note may have been deleted since indexing — leave text empty
    }
  }

  for (const r of results) {
    const body = bodies.get(r.noteId);
    if (!body) continue;
    for (const b of r.blocks) {
      if (b.bodyIdx >= 0 && b.bodyIdx + b.length <= body.length) {
        b.text = body.substring(b.bodyIdx, b.bodyIdx + b.length);
      }
    }
  }
}

// ── command registration ────────────────────────────────────────────

export async function register_api_commands(runtime: ApiRuntime): Promise<void> {

  // ── jarvis.api.status ───────────────────────────────────────────
  await joplin.commands.register({
    name: 'jarvis.api.status',
    label: 'Jarvis API: Status',
    execute: async () => {
      const ready = isReady(runtime);
      const modelId = runtime.model_embed.id ?? null;
      let indexStats: { noteCount: number; blockCount: number } | null = null;

      if (ready && modelId) {
        const ms = getModelStats(modelId);
        const cache = corpusCaches.get(modelId);
        const cacheStats = cache?.getStats();
        indexStats = {
          noteCount: ms?.noteCount ?? 0,
          blockCount: cacheStats?.blocks ?? ms?.rowCount ?? 0,
        };
      }

      return { ok: true, version: API_VERSION, ready, modelId, indexStats };
    },
  });

  // ── jarvis.api.search ───────────────────────────────────────────
  await joplin.commands.register({
    name: 'jarvis.api.search',
    label: 'Jarvis API: Search',
    execute: async (...args: any[]) => {
      const params = (args[0] && typeof args[0] === 'object') ? args[0] : {};
      const { query, noteId, limit, minSimilarity } = params as {
        query?: string;
        noteId?: string;
        limit?: number;
        minSimilarity?: number;
      };

      // Validate: at least one of query / noteId required
      if ((!query || typeof query !== 'string' || !query.trim()) &&
          (!noteId || typeof noteId !== 'string' || !noteId.trim())) {
        return errorResponse('invalid_input', 'Provide at least one of: query (string), noteId (string)');
      }

      if (!isReady(runtime)) {
        return errorResponse('not_ready', 'Jarvis embedding index is not initialised');
      }

      // Build overrides for limit / minSimilarity
      const searchSettings = { ...runtime.settings };
      if (typeof limit === 'number' && limit > 0) {
        searchSettings.notes_max_hits = limit;
      }
      if (typeof minSimilarity === 'number' && minSimilarity >= 0 && minSimilarity <= 1) {
        searchSettings.notes_min_similarity = minSimilarity;
      }

      try {
        // Resolve query text and note context
        let searchQuery: string;
        let searchId: string;
        let searchTitle: string;
        let markupLanguage: number;

        if (noteId) {
          // Fetch the note to use as query context
          const note = await joplin.data.get(['notes', noteId],
            { fields: ['id', 'title', 'body', 'markup_language'] });
          searchId = note.id;
          searchTitle = note.title ?? '';
          markupLanguage = note.markup_language ?? 1;
          searchQuery = query?.trim() || note.body || '';
          clearApiResponse(note);
        } else {
          searchId = '__jarvis_api__';
          searchTitle = '';
          markupLanguage = 1;  // Markdown
          searchQuery = query!.trim();
        }

        const results = await find_nearest_notes(
          [],  // no pre-loaded embeddings — let the function use the cache / compute on the fly
          searchId,
          markupLanguage,
          searchTitle,
          searchQuery,
          runtime.model_embed,
          searchSettings,
          true,   // return_grouped_notes
          undefined,  // panel (no progress bar)
          false,  // isUpdateInProgress
        );

        const serialised = toSerializableResults(results);
        await attachBlockText(serialised);

        return { ok: true, results: serialised };
      } catch (err) {
        log.error('jarvis.api.search failed', err);
        return errorResponse('search_failed', String(err));
      }
    },
  });
}
