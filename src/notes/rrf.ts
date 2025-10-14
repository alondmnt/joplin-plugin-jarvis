export interface RrfItem {
  id: string;
  score: number;
  [key: string]: unknown;
}

export interface RrfQueryResults {
  label: string;
  items: RrfItem[];
}

export interface RrfOptions {
  k?: number;
  maxResults?: number;
  logTopN?: number;
}

export interface RrfSourceContribution {
  query: string;
  rank: number;
  reciprocal: number;
  score: number;
}

export interface RrfResult {
  id: string;
  fusedScore: number;
  firstRank: number;
  contributions: RrfSourceContribution[];
}

export interface RrfFusionLog {
  query: string;
  hits: number;
  topIds: string[];
}

export interface RrfFusionOutput {
  results: RrfResult[];
  logs: RrfFusionLog[];
}

const DEFAULT_K = 60;
const DEFAULT_MAX_LOG_IDS = 5;

/**
 * Fuse multiple ranked lists using Reciprocal Rank Fusion with deterministic ties.
 * Also emits lightweight logs so callers can inspect per-query contributions.
 */
export function reciprocalRankFusion(
  queries: RrfQueryResults[],
  options: RrfOptions = {},
): RrfFusionOutput {
  const k = options.k ?? DEFAULT_K;
  const maxResults = options.maxResults ?? Number.POSITIVE_INFINITY;
  const logTopN = options.logTopN ?? DEFAULT_MAX_LOG_IDS;

  const accum = new Map<
    string,
    {
      score: number;
      firstRank: number;
      contributions: RrfSourceContribution[];
    }
  >();

  const logs: RrfFusionLog[] = [];

  for (const query of queries) {
    const seenIds = new Set<string>();
    let hits = 0;
    for (let index = 0; index < query.items.length; index += 1) {
      const item = query.items[index];
      if (!item || !item.id) continue;
      if (seenIds.has(item.id)) continue;
      seenIds.add(item.id);
      hits += 1;

      const reciprocal = 1 / (k + index + 1);
      const existing = accum.get(item.id);
      const contribution: RrfSourceContribution = {
        query: query.label,
        rank: index + 1,
        reciprocal,
        score: item.score,
      };

      if (existing) {
        existing.score += reciprocal;
        existing.contributions.push(contribution);
        if (index + 1 < existing.firstRank) {
          existing.firstRank = index + 1;
        }
      } else {
        accum.set(item.id, {
          score: reciprocal,
          firstRank: index + 1,
          contributions: [contribution],
        });
      }
    }

    logs.push({
      query: query.label,
      hits,
      topIds: query.items.slice(0, logTopN).map((item) => item.id),
    });
  }

  const results: RrfResult[] = Array.from(accum.entries()).map(([id, value]) => ({
    id,
    fusedScore: value.score,
    firstRank: value.firstRank,
    contributions: value.contributions,
  }));

  results.sort((a, b) => {
    if (b.fusedScore !== a.fusedScore) {
      return b.fusedScore - a.fusedScore;
    }
    if (a.firstRank !== b.firstRank) {
      return a.firstRank - b.firstRank;
    }
    return a.id.localeCompare(b.id);
  });

  const limitedResults = Number.isFinite(maxResults) ? results.slice(0, maxResults) : results;

  return {
    results: limitedResults,
    logs,
  };
}
