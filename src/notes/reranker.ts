import type { TextGenerationModel } from '../models/models';
import type { ChatNotesPlanResponse, ChatNotesRerankResponse } from '../prompts/chatWithNotes';
import { buildChatNotesRerankPrompt, validateChatNotesRerankResponse } from '../prompts/chatWithNotes';
import type { LexicalCandidate } from './lexicalRetrieval';

export interface PairwiseRerankOptions {
  pairwisePoolSize: number;
  topK: number;
  pairwiseCap: number;
  skipDelta01: number;
  skipDelta12: number;
  abortSignal?: AbortSignal;
}

export interface RerankComparisonLog {
  pair: [string, string];
  winner: 'A' | 'B' | 'none';
  confident: 0 | 1;
  reason?: string;
  raw?: string;
  promptVersion: string;
}

export interface PairwiseRerankResult {
  ordered: LexicalCandidate[];
  skipped: boolean;
  comparisons: RerankComparisonLog[];
  normalizedScores: number[];
}

function extractJsonObject(raw: string): string {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Reranker response missing JSON object');
  }
  return raw.slice(start, end + 1);
}

function normalizeScores(candidates: LexicalCandidate[]): number[] {
  if (candidates.length === 0) return [];
  const scores = candidates.map((candidate) => candidate.score0);
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  if (maxScore === minScore) {
    return candidates.map(() => 1);
  }
  return scores.map((score) => (score - minScore) / (maxScore - minScore));
}

function shouldSkipRerank(normalizedScores: number[], options: PairwiseRerankOptions): boolean {
  if (normalizedScores.length < 3) return true;
  const delta01 = normalizedScores[0] - normalizedScores[1];
  const delta12 = normalizedScores[1] - normalizedScores[2];
  return delta01 >= options.skipDelta01 && delta12 >= options.skipDelta12;
}

async function runComparison(
  model: TextGenerationModel,
  plan: ChatNotesPlanResponse,
  candidateA: LexicalCandidate,
  candidateB: LexicalCandidate,
  abortSignal?: AbortSignal,
): Promise<{ response: ChatNotesRerankResponse; raw: string }> {
  const prompt = buildChatNotesRerankPrompt({
    query: plan.normalized_query,
    normalizedQuery: plan.normalized_query,
    candidateA: {
      id: candidateA.id,
      title: candidateA.title,
      snippet: candidateA.snippet,
      score0: candidateA.score0,
      headings: candidateA.headings,
      updated_at: candidateA.updatedAt ?? undefined,
    },
    candidateB: {
      id: candidateB.id,
      title: candidateB.title,
      snippet: candidateB.snippet,
      score0: candidateB.score0,
      headings: candidateB.headings,
      updated_at: candidateB.updatedAt ?? undefined,
    },
  });

  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const repairSuffix = attempt === 0
      ? ''
      : `
The previous response was invalid JSON (${String(lastError).slice(0, 120)}). Respond again with VALID JSON only.`;
    const raw = await model.chat(`${prompt}${repairSuffix}`, false, abortSignal);
    try {
      const payload = extractJsonObject(raw);
      const parsed = JSON.parse(payload);
      const response = validateChatNotesRerankResponse(parsed);
      return { response, raw };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Reranker failed with unknown error');
}

/**
 * Run the deterministic pairwise tournament over the lexical candidate pool.
 * Applies skip rules, JSON contract enforcement, and stable ordering guarantees.
 */
export async function runPairwiseRerank(
  model: TextGenerationModel,
  plan: ChatNotesPlanResponse,
  candidates: LexicalCandidate[],
  options: PairwiseRerankOptions,
): Promise<PairwiseRerankResult> {
  if (candidates.length === 0) {
    return {
      ordered: [],
      skipped: true,
      comparisons: [],
      normalizedScores: [],
    };
  }

  const poolSize = Math.min(options.pairwisePoolSize, candidates.length);
  const pool = candidates.slice(0, poolSize).map((candidate) => candidate);
  const normalizedScores = normalizeScores(pool);

  if (shouldSkipRerank(normalizedScores, options)) {
    return {
      ordered: candidates.slice(0, options.topK),
      skipped: true,
      comparisons: [],
      normalizedScores,
    };
  }

  const logs: RerankComparisonLog[] = [];
  let comparisonsMade = 0;

  outer: for (let i = 0; i < pool.length; i += 1) {
    for (let j = i + 1; j < pool.length; j += 1) {
      if (comparisonsMade >= options.pairwiseCap) break outer;
      const candidateA = pool[i];
      const candidateB = pool[j];
      try {
        const { response, raw } = await runComparison(model, plan, candidateA, candidateB, options.abortSignal);
        comparisonsMade += 1;
        logs.push({
          pair: [candidateA.id, candidateB.id],
          winner: response.winner,
          confident: response.confident,
          reason: response.reason,
          raw,
          promptVersion: response.prompt_version,
        });

        if (response.winner === 'B') {
          pool.splice(j, 1);
          pool.splice(i, 0, candidateB);
        } else if (response.winner === 'none') {
          // Retain lexical order (candidateA before candidateB).
        } else if (response.winner === 'A') {
          // Already in lexical order (candidateA before candidateB).
        }
      } catch (error) {
        console.warn('Pairwise rerank comparison failed; retaining lexical order', error);
        logs.push({
          pair: [candidateA.id, candidateB.id],
          winner: 'none',
          confident: 0,
          promptVersion: 'chat-notes-rerank-2',
          reason: `error:${String(error).slice(0, 80)}`,
        });
        comparisonsMade += 1;
      }
    }
  }

  const reranked = pool.concat(candidates.slice(poolSize));
  const ordered = reranked.slice(0, options.topK);

  return {
    ordered,
    skipped: false,
    comparisons: logs,
    normalizedScores,
  };
}
