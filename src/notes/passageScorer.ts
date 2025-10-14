import { normalizeText } from './tokenizer';

export interface Bm25lParams {
  k1: number;
  b: number;
  delta: number;
}

export const DEFAULT_BM25L_PARAMS: Bm25lParams = {
  k1: 1.5,
  b: 0.75,
  delta: 0.5,
};

export interface Bm25lScoreInput {
  termFrequencies: Record<string, number>;
  documentLength: number;
  averageDocumentLength: number;
  totalDocuments: number;
  documentFrequencies: Record<string, number>;
  queryTerms: string[];
  params?: Bm25lParams;
}

function calculateIdf(totalDocuments: number, documentFrequency: number): number {
  if (totalDocuments <= 0) return 0;
  const df = Math.max(0, Math.min(documentFrequency, totalDocuments));
  if (df >= totalDocuments) return 0;
  const numerator = totalDocuments - df + 0.5;
  const denominator = df + 0.5;
  if (denominator <= 0) return 0;
  const ratio = numerator / denominator;
  if (ratio <= 0) return 0;
  const raw = Math.log(1 + ratio);
  return raw > 0 ? raw : 0;
}

/**
 * Calculate BM25L score for a passage given query term stats and corpus stats.
 * We operate on pre-tokenized lowercase terms to stay deterministic across platforms.
 */
export function computeBm25LScore(input: Bm25lScoreInput): number {
  const params = input.params ?? DEFAULT_BM25L_PARAMS;
  const {
    termFrequencies,
    documentLength,
    averageDocumentLength,
    totalDocuments,
    documentFrequencies,
    queryTerms,
  } = input;

  if (
    documentLength <= 0 ||
    averageDocumentLength <= 0 ||
    Number.isNaN(documentLength) ||
    Number.isNaN(averageDocumentLength)
  ) {
    return 0;
  }

  const lengthNorm = (termFrequency: number): number => {
    const denom = (1 - params.b) + params.b * (documentLength / averageDocumentLength);
    const safeDenom = denom > 0 ? denom : 1;
    return termFrequency / safeDenom;
  };

  let score = 0;

  const uniqueQueryTerms = Array.from(new Set(queryTerms.map((term) => term.toLowerCase())));

  for (const term of uniqueQueryTerms) {
    const tf = termFrequencies[term] ?? termFrequencies[term.toLowerCase()] ?? 0;
    if (tf <= 0) continue;

    const idf = calculateIdf(totalDocuments, documentFrequencies[term] ?? 0);
    if (idf <= 0) continue;

    const normalizedTf = lengthNorm(tf);
    const tfPrime = normalizedTf + params.delta;
    const numerator = (params.k1 + 1) * tfPrime;
    const denominator = params.k1 + tfPrime;
    if (denominator <= 0) continue;

    score += idf * (numerator / denominator);
  }

  return score;
}

export interface SpanProximityOptions {
  hardTerms: Iterable<string>;
  allTerms: Iterable<string>;
}

/**
 * Reward windows that keep multiple query terms close together.
 * Uses a sliding window over token hits with separate handling for hard terms.
 */
export function spanProximityScore(tokens: string[], options: SpanProximityOptions): number {
  const hardTermSet = new Set(Array.from(options.hardTerms).map((term) => term.toLowerCase()));
  const allTermSet = new Set(Array.from(options.allTerms).map((term) => term.toLowerCase()));

  const targetTerms = hardTermSet.size >= 2 ? hardTermSet : allTermSet;
  if (targetTerms.size < 2) return 0;

  type Hit = { index: number; term: string };
  const hits: Hit[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index].toLowerCase();
    if (targetTerms.has(token)) {
      hits.push({ index, term: token });
    }
  }

  if (hits.length < 2) return 0;

  const counts = new Map<string, number>();
  let uniqueTerms = 0;
  let left = 0;
  let bestWindow = Number.POSITIVE_INFINITY;

  for (let right = 0; right < hits.length; right += 1) {
    const rightTerm = hits[right].term;
    const existing = counts.get(rightTerm) ?? 0;
    if (existing === 0) uniqueTerms += 1;
    counts.set(rightTerm, existing + 1);

    while (uniqueTerms >= 2 && left <= right) {
      const windowSize = hits[right].index - hits[left].index + 1;
      if (windowSize < bestWindow) {
        bestWindow = windowSize;
      }
      const leftTerm = hits[left].term;
      const leftCount = (counts.get(leftTerm) ?? 0) - 1;
      counts.set(leftTerm, leftCount);
      if (leftCount <= 0) {
        counts.delete(leftTerm);
        uniqueTerms -= 1;
      }
      left += 1;
    }
  }

  if (!Number.isFinite(bestWindow) || bestWindow <= 1) {
    return Number.isFinite(bestWindow) && bestWindow === 1 ? 1 : 0;
  }

  return 2 / bestWindow;
}

/**
 * Count how many quoted phrases appear verbatim in the given text.
 * Normalizes both sides so we can tolerate Unicode punctuation variants.
 */
export function anchorQuoteMatchCount(text: string, quotedPhrases: Iterable<string>): number {
  if (!text) return 0;
  const normalizedText = normalizeText(text);
  let count = 0;
  for (const phrase of quotedPhrases) {
    const normalizedPhrase = normalizeText(phrase);
    if (!normalizedPhrase) continue;
    if (normalizedText.includes(normalizedPhrase)) {
      count += 1;
    }
  }
  return count;
}

/**
 * Count distinct query terms that appear anywhere along the heading path.
 * Used as an additive boost for notes/headings that align structurally with the query.
 */
export function headingPathMatchCount(headingPath: string[], queryTerms: Iterable<string>): number {
  if (!headingPath.length) return 0;
  const normalizedHeadings = headingPath.map((heading) => normalizeText(heading));
  const matches = new Set<string>();
  for (const term of queryTerms) {
    const normalizedTerm = normalizeText(term);
    if (!normalizedTerm) continue;
    if (normalizedHeadings.some((heading) => heading.includes(normalizedTerm))) {
      matches.add(normalizedTerm);
    }
  }
  return matches.size;
}

/**
 * Simple binary boost when a note is part of the conversational context set.
 */
export function contextNoteBoost(noteId: string, contextNotes: Set<string>): number {
  if (!noteId) return 0;
  return contextNotes.has(noteId) ? 1 : 0;
}

/**
 * Compute a linear decay freshness signal over the configured window.
 * Returns 1 for up-to-date notes, 0 when outside the window or invalid dates.
 */
export function recencyBoost(
  updatedAtIso: string | null | undefined,
  referenceDate: Date = new Date(),
  freshnessWindowDays = 30,
): number {
  if (!updatedAtIso) return 0;
  const updatedAt = new Date(updatedAtIso);
  if (Number.isNaN(updatedAt.getTime())) return 0;

  const diffMs = referenceDate.getTime() - updatedAt.getTime();
  if (diffMs <= 0) return 1;

  const diffDays = diffMs / (1000 * 60 * 60 * 24);
  if (diffDays >= freshnessWindowDays) return 0;

  return 1 - diffDays / freshnessWindowDays;
}

export interface Window {
  start: number;
  end: number;
}

export interface MmrCandidate<T> {
  id: string;
  score: number;
  window: Window;
  payload: T;
}

export interface MmrOptions {
  lambda?: number;
  maxSelections?: number;
  maxOverlapRatio?: number;
}

/**
 * Apply Maximal Marginal Relevance to keep diverse, high-scoring windows.
 * Balances relevance and overlap using the configured lambda/overlap thresholds.
 */
export function selectWithMmr<T>(candidates: MmrCandidate<T>[], options: MmrOptions = {}): MmrCandidate<T>[] {
  if (candidates.length === 0) return [];

  const lambda = options.lambda ?? 0.7;
  const maxSelections = options.maxSelections ?? candidates.length;
  const maxOverlapRatio = options.maxOverlapRatio ?? 0.8;

  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const selected: MmrCandidate<T>[] = [];

  while (selected.length < maxSelections && selected.length < sorted.length) {
    let bestCandidate: MmrCandidate<T> | null = null;
    let bestScore = -Infinity;

    for (const candidate of sorted) {
      if (selected.includes(candidate)) continue;

      const relevance = lambda * candidate.score;

      let maxOverlap = 0;
      for (const chosen of selected) {
        const overlapStart = Math.max(candidate.window.start, chosen.window.start);
        const overlapEnd = Math.min(candidate.window.end, chosen.window.end);
        const overlap = Math.max(0, overlapEnd - overlapStart);
        const candidateSpan = Math.max(1, candidate.window.end - candidate.window.start);
        const overlapRatio = overlap / candidateSpan;
        if (overlapRatio > maxOverlap) {
          maxOverlap = overlapRatio;
        }
      }

      if (maxOverlap >= maxOverlapRatio) continue;

      const redundancy = (1 - lambda) * maxOverlap;
      const mmrScore = relevance - redundancy;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestCandidate = candidate;
      }
    }

    if (!bestCandidate) break;
    selected.push(bestCandidate);
  }

  return selected;
}
