import { ChatNotesPlanResponse } from '../prompts/chatWithNotes';
import { tokenizeForSearch, containsCjkOrThai } from './tokenizer';
import {
  buildAcronym,
  extractAcronyms,
  extractCapitalizedEntities,
  extractTopKeywords,
  sanitizeQueryText,
} from './textUtils';

export interface FallbackPlanOptions {
  maxHardTerms?: number;     // default 2
  maxSoftTerms?: number;     // default 8
  maxEntities?: number;      // default 5
  maxExpansions?: number;    // default 10
  maxAcronyms?: number;      // default 8
  extraAcronymPhrases?: string[];
  maxNormalizedChars?: number; // default 80
}

function uniqPreserve<T>(arr: T[]): T[] {
  const seen = new Set<T>();
  const out: T[] = [];
  for (const x of arr) if (!seen.has(x)) { seen.add(x); out.push(x); }
  return out;
}

function deriveAcronyms(text: string, phrases: string[] = [], maxAcronyms = 8): string[] {
  const set = new Set<string>();
  for (const a of extractAcronyms(text)) set.add(a);
  for (const p of phrases) {
    const cand = buildAcronym(p);
    if (cand) set.add(cand);
  }
  // filter: ≥2 chars, ≥2 letters
  const ok = Array.from(set).filter(a => a.length >= 2 && /[A-Za-z].*[A-Za-z]/.test(a));
  return ok.slice(0, maxAcronyms);
}

function clip(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n);
}

/**
 * Construct a deterministic lexical plan when the planner LLM fails.
 * Relies on lightweight heuristics (keywords/entities/acronyms) so Step B can continue.
 */
export function buildFallbackPlanFromText(
  text: string,
  options: FallbackPlanOptions = {},
): ChatNotesPlanResponse {
  const maxHard = options.maxHardTerms ?? 2;
  const maxSoft = options.maxSoftTerms ?? 8;
  const maxEnt  = options.maxEntities ?? 5;
  const maxExp  = options.maxExpansions ?? 10;
  const maxAcr  = options.maxAcronyms ?? 8;
  const maxNorm = options.maxNormalizedChars ?? 80;

  const cleaned = sanitizeQueryText(text);
  const { hard: hardRaw, soft: softRaw } = extractTopKeywords(cleaned, maxHard, maxSoft);

  // Dedup and separate hard/soft (soft = softRaw - hardRaw)
  const hard_terms = uniqPreserve(hardRaw).slice(0, maxHard);
  const soft_terms = uniqPreserve(softRaw.filter(t => !hard_terms.includes(t))).slice(0, maxSoft);

  // Expansions: start with soft terms (we keep synonyms/aliases here if any heuristic added later)
  const expansions = uniqPreserve(soft_terms).slice(0, maxExp);

  // Entities from original (case-sensitive) text; cap and dedup
  const entities = uniqPreserve(extractCapitalizedEntities(text, maxEnt)).slice(0, maxEnt);

  // Acronyms from text + entity phrases
  const acronyms = deriveAcronyms(text, entities, maxAcr);

  // Tokenization (for last-resort normalized query)
  const toks = tokenizeForSearch(cleaned);

  // normalized_query selection (avoid CJK bigrams)
  let normalized = cleaned.trim();
  if (!normalized) {
    if (!containsCjkOrThai(cleaned) && toks.tokens.length) {
      normalized = toks.tokens.slice(0, 3).join(' ');
    } else if (hard_terms.length) {
      normalized = hard_terms.join(' ');
    } else {
      normalized = 'notes';
    }
  }
  normalized = clip(normalized, maxNorm);

  return {
    prompt_version: 'chat-notes-plan-2',
    normalized_query: normalized,
    expansions,
    hard_terms,
    soft_terms,
    entities,
    acronyms,
    filters: {
      tags: [],
      notebooks: [],
      created_after: null,
      created_before: null,
    },
    context_notes: [],
  };
}
