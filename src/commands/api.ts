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
import { clearApiResponse } from '../utils';
import { find_nearest_notes, group_by_notes, preprocess_note_for_hashing, convert_newlines, corpusCaches, userDataStore } from '../notes/embeddings';
import { read_user_data_embeddings } from '../notes/userDataReader';
import { maxsim_search, keyword_rerank } from '../notes/hybridSearch';
import type { NoteEmbedding, BlockEmbedding } from '../notes/embeddings';
import type { TextEmbeddingModel } from '../models/models';
import type { JarvisSettings } from '../ux/settings';
import { getModelStats } from '../notes/modelStats';

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
      // Reconstruct the canonical body using the same pipeline as indexing:
      // preprocess_note_for_hashing (HTML-to-text + OCR append + strip Jarvis blocks)
      // then convert_newlines (\r\n → \n) to match the offsets in body_idx.
      const note: any = { id, body: resp.body ?? '', markup_language: resp.markup_language };
      await preprocess_note_for_hashing(note);
      note.body = convert_newlines(note.body);
      bodies.set(id, note.body);
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
        let results: NoteEmbedding[] | undefined;

        if (noteId) {
          // note-based search: multi-chunk MaxSim (same as panel)
          const note = await joplin.data.get(['notes', noteId],
            { fields: ['id', 'title', 'body', 'markup_language'] });
          const noteTitle = note.title ?? '';
          const noteBody = note.body ?? '';
          const noteMarkup = note.markup_language ?? 1;
          clearApiResponse(note);

          // load note's chunk embeddings
          let query_chunks: BlockEmbedding[] = runtime.model_embed.embeddings.filter(b => b.id === noteId);
          if (query_chunks.length === 0 && searchSettings.notes_db_in_user_data) {
            const loaded = await read_user_data_embeddings({
              store: userDataStore, modelId: runtime.model_embed.id, noteIds: [noteId],
            });
            if (loaded.length > 0) { query_chunks = loaded[0].blocks; }
          }

          if (query_chunks.length > 0 && searchSettings.notes_multi_chunk_search) {
            // multi-chunk MaxSim + keyword rerank on title
            const query_embeddings = query_chunks.map(c => c.embedding);
            const cache = corpusCaches.get(runtime.model_embed.id);
            const scored = maxsim_search(query_embeddings, runtime.model_embed.embeddings, cache, noteId, searchSettings);
            if (scored.length > 0) {
              const reranked = noteTitle
                ? await keyword_rerank(scored, [noteTitle], searchSettings)
                : scored;
              results = await group_by_notes(reranked, searchSettings);
            }
          }

          if (!results) {
            // fallback: single-vector search using query text or note body
            const searchQuery = query?.trim() || noteBody;
            const flat = await find_nearest_notes(
              [], noteId, noteMarkup, noteTitle, searchQuery,
              runtime.model_embed, searchSettings, false,
              undefined, false, undefined, true,
            );
            const blocks = (flat.length > 0 && flat[0].embeddings.length > 0)
              ? await keyword_rerank(flat[0].embeddings, [query?.trim() || noteTitle].filter(Boolean), searchSettings)
              : [];
            results = blocks.length > 0
              ? await group_by_notes(blocks, searchSettings)
              : flat;
          }
        } else {
          // text query search: same as search box (flat + keyword rerank + group)
          const searchQuery = query!.trim();
          const flat = await find_nearest_notes(
            [], '__jarvis_api__', 1, '', searchQuery,
            runtime.model_embed, searchSettings, false,
            undefined, false, undefined, true,
          );
          const blocks = (flat.length > 0 && flat[0].embeddings.length > 0)
            ? await keyword_rerank(flat[0].embeddings, [searchQuery], searchSettings)
            : [];
          results = blocks.length > 0
            ? await group_by_notes(blocks, searchSettings)
            : flat;
        }

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
