/**
 * LLM query decomposition for hybrid search.
 *
 * Uses the chat model to decompose a user query into focused sub-queries,
 * each with a semantic component (for embedding search) and keyword terms
 * (for Joplin keyword search).
 */
import { TextGenerationModel } from '../models/models';
import { with_timeout } from '../utils';
import { getLogger } from '../utils/logger';

const log = getLogger();

/**
 * Decompose a user query into focused sub-queries for hybrid search.
 *
 * @param query - the user's question
 * @param model_gen - text generation model for LLM call
 * @returns array of sub-queries, or null on failure/timeout
 */
export async function decompose_query(
  query: string,
  model_gen: TextGenerationModel,
): Promise<{semantic: string, keywords: string[]}[] | null> {
  const prompt = `Decompose this question into 1-3 focused search sub-queries.
For each, output on a separate line:
SEARCH: <semantic query> | KEYWORDS: <terms or NONE>

Rules:
- Use 1 sub-query when the question targets a single topic, entity, or time period.
- Use 2-3 for comparisons (one sub-query per side) or multiple distinct entities/topics.
- Use "quoted phrases" for compound terms. Drop evaluative words from keywords.
- Combine co-occurring entities in a single keyword term.

Question: ${query}`;

  try {
    // use temperature 0 for deterministic decomposition
    const saved_temperature = model_gen.temperature;
    model_gen.temperature = 0;
    let response: string;
    try {
      response = await with_timeout(10_000, model_gen.complete(prompt));
    } finally {
      model_gen.temperature = saved_temperature;
    }
    if (!response) { return null; }

    const results: {semantic: string, keywords: string[]}[] = [];
    for (const line of response.split('\n')) {
      const match = line.match(/SEARCH:\s*(.+?)\s*\|\s*KEYWORDS:\s*(.+)/i);
      if (!match) { continue; }

      const semantic = match[1].trim();
      if (!semantic) { continue; }

      const kw_raw = match[2].trim();
      const keywords = kw_raw.toLowerCase() === 'none'
        ? []
        : kw_raw.split(',').map(k => k.trim()).filter(k => k.length > 0);

      results.push({ semantic, keywords });
    }

    return results.length > 0 ? results : null;
  } catch (error) {
    log.info(`[Hybrid] decomposition failed: ${error.message || error}`);
    return null;
  }
}
