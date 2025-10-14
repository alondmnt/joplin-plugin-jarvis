import { normalizeText, tokenizeForSearch } from './tokenizer';

const WORD_RE = /[A-Za-z0-9]+/g;
const CAPITALIZED_WORD_RE = /\b[A-Z][a-zA-Z0-9]+\b/g;
const ACRONYM_RE = /\b[A-Z]{2,}\b/g;

/**
 * Build an uppercase acronym from the first letters of alphanumeric words.
 */
export function buildAcronym(phrase: string): string {
  if (!phrase) return '';
  const matches = phrase.match(WORD_RE);
  if (!matches) return '';
  const letters = matches
    .map((word) => word[0])
    .filter((char) => /[A-Za-z]/.test(char));
  if (letters.length < 2) return '';
  return letters.join('').toUpperCase();
}

/**
 * Extract existing acronyms (≥2 capital letters) from the given text.
 */
export function extractAcronyms(text: string): string[] {
  if (!text) return [];
  const acronyms = text.match(ACRONYM_RE) ?? [];
  return Array.from(new Set(acronyms));
}

/**
 * Grab distinct capitalized tokens to approximate named entities.
 * We cap the result to avoid flooding the planner fallback.
 */
export function extractCapitalizedEntities(text: string, limit = 5): string[] {
  if (!text) return [];
  const matches = text.match(CAPITALIZED_WORD_RE) ?? [];
  const lowerSeen = new Set<string>();
  const entities: string[] = [];
  for (const match of matches) {
    const lower = match.toLowerCase();
    if (lowerSeen.has(lower)) continue;
    lowerSeen.add(lower);
    entities.push(match);
    if (entities.length >= limit) break;
  }
  return entities;
}

export interface KeywordExtractionResult {
  hard: string[];
  soft: string[];
}

/**
 * Use simple frequency counts over lexical tokens to pick hard/soft keywords.
 * Hard keywords are the top-k tokens; soft keywords are the next best candidates.
 */
export function extractTopKeywords(text: string, hardLimit = 2, softLimit = 8): KeywordExtractionResult {
  if (!text) {
    return { hard: [], soft: [] };
  }

  const { tokens } = tokenizeForSearch(text);
  const counts = new Map<string, number>();
  for (const token of tokens) {
    if (token.length < 3) continue;
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  const sorted = Array.from(counts.entries())
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    })
    .map(([token]) => token);

  const hard = sorted.slice(0, hardLimit);
  const soft = sorted.slice(hardLimit, hardLimit + softLimit);
  return { hard, soft };
}

/**
 * Normalize and trim text for building lexical queries.
 */
export function sanitizeQueryText(value: string): string {
  return normalizeText(value).trim();
}
