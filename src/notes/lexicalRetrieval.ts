import joplin from 'api';
import type { ChatNotesPlanResponse } from '../prompts/chatWithNotes';
import { reciprocalRankFusion } from './rrf';
import type { PlanQueries, QueryLabel } from './queryBuilder';
import { normalizeText, tokenizeForSearch } from './tokenizer';
import {
  computeBm25LScore,
  spanProximityScore,
  headingPathMatchCount,
  contextNoteBoost,
  recencyBoost,
} from './passageScorer';
import type { RrfFusionLog, RrfResult, RrfSourceContribution } from './rrf';
import { phraseMatch } from './phraseMatcher';

export interface LexicalRetrievalConfig {
  candidateLimit: number;
  recencyWindowDays?: number;
  highlightRadius?: number;
}

export interface LexicalFeatureScores {
  bm25: number;
  headingBoost: number;
  phraseProximity: number;
  entityOverlap: number;
  tagNotebookOverlap: number;
  recencyBoost: number;
  prevContextBoost: number;
  titleMatch: number;
}

export interface LexicalCandidate {
  id: string;
  title: string;
  body: string;
  snippet: string;
  headings: string[];
  updatedAt: string | null;
  notebookPath: string | null;
  tags: string[];
  fusedScore: number;
  firstRank: number;
  score0: number;
  features: LexicalFeatureScores;
  contributions: RrfSourceContribution[];
  tokens: string[];
  normalizedBody: string;
}

export interface LexicalRetrievalResult {
  candidates: LexicalCandidate[];
  rrfLogs: RrfFusionLog[];
}

interface RawSearchHit {
  id: string;
  title: string;
  parent_id?: string;
  user_updated_time?: number;
  body?: string;
  excerpt?: string;
  highlight?: string;
}

interface NoteDetails {
  id: string;
  title: string;
  body: string;
  parent_id: string | null;
  user_updated_time: number | null;
  tags: string[];
}

const DEFAULT_RECENCY_WINDOW = 30;
const DEFAULT_HIGHLIGHT_RADIUS = 240;
const SEARCH_PAGE_SIZE = 50;

function toIso(ms: number | undefined | null): string | null {
  if (!ms && ms !== 0) return null;
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString();
}

function dedupeStrings(values: Iterable<string>, transform: (value: string) => string = (value) => value) {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const transformed = transform(value);
    if (!transformed) continue;
    if (seen.has(transformed)) continue;
    seen.add(transformed);
    out.push(transformed);
  }
  return out;
}

async function fetchSearchResults(query: string, limit: number): Promise<RawSearchHit[]> {
  const results: RawSearchHit[] = [];
  let page = 1;
  while (results.length < limit) {
    const response = await joplin.data.get(['search'], {
      query,
      limit: Math.min(SEARCH_PAGE_SIZE, limit - results.length),
      page,
      type: 'note',
      fields: ['id', 'title', 'parent_id', 'body', 'user_updated_time'],
      highlight: 1,
    });

    const items = Array.isArray(response?.items) ? response.items : [];
    results.push(...(items as RawSearchHit[]));
    if (!response?.has_more) break;
    page += 1;
  }
  return results.slice(0, limit);
}

async function fetchNoteDetails(noteId: string): Promise<NoteDetails | null> {
  try {
    const note = await joplin.data.get(['notes', noteId], {
      fields: ['id', 'title', 'body', 'parent_id', 'user_updated_time'],
    });
    if (!note) return null;
    const tagResponse = await joplin.data.get(['notes', noteId, 'tags'], {
      fields: ['title'],
    });
    const tags = Array.isArray(tagResponse?.items)
      ? (tagResponse.items as Array<{ title: string }>).map((item) => item.title ?? '').filter(Boolean)
      : [];
    return {
      id: note.id,
      title: note.title ?? '',
      body: note.body ?? '',
      parent_id: note.parent_id ?? null,
      user_updated_time: typeof note.user_updated_time === 'number' ? note.user_updated_time : null,
      tags,
    };
  } catch (error) {
    console.warn('Failed to fetch note details', noteId, error);
    return null;
  }
}

interface FolderDetails {
  id: string;
  title: string;
  parent_id: string | null;
}

const notebookPathCache = new Map<string, string | null>();

async function resolveNotebookPath(folderId: string | null): Promise<string | null> {
  if (!folderId) return null;
  if (notebookPathCache.has(folderId)) {
    return notebookPathCache.get(folderId) ?? null;
  }
  try {
    const folder: FolderDetails | null = await joplin.data.get(['folders', folderId], {
      fields: ['id', 'title', 'parent_id'],
    });
    if (!folder) {
      notebookPathCache.set(folderId, null);
      return null;
    }
    const parentPath = await resolveNotebookPath(folder.parent_id ?? null);
    const path = parentPath ? `${parentPath}/${folder.title ?? ''}` : folder.title ?? '';
    const normalisedPath = path ? path : null;
    notebookPathCache.set(folderId, normalisedPath);
    return normalisedPath;
  } catch (error) {
    console.warn('Failed to resolve notebook path', folderId, error);
    notebookPathCache.set(folderId, null);
    return null;
  }
}

function extractSnippet(note: NoteDetails, highlight: string | undefined, radius: number): string {
  if (highlight && typeof highlight === 'string' && highlight.trim()) {
    const plain = highlight
      .replace(/<\/?mark>/gi, '')
      .replace(/<\/?[^>]+>/g, '')
      .trim();
    if (plain) return plain;
  }

  const body = note.body ?? '';
  if (!body) return '';
  const trimmed = body.trim();
  if (trimmed.length <= radius) return trimmed;

  return `${trimmed.slice(0, radius)}…`;
}

function extractHeadings(body: string): string[] {
  if (!body) return [];
  const headings: string[] = [];
  const lines = body.split('\n');
  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.*)$/);
    if (match) {
      headings.push(match[2].trim());
    }
  }
  return headings;
}

/**
 * Build the full token vocabulary derived from the planner output.
 * The result powers BM25L, proximity, and heading/tag matching so it must
 * include everything the planner surfaced (normalized query, terms, expansions).
 */
export function buildQueryTokenSet(plan: ChatNotesPlanResponse): Set<string> {
  const tokens = new Set<string>();
  const addTokens = (text: string) => {
    if (!text) return;
    const result = tokenizeForSearch(text);
    for (const token of result.tokens) {
      tokens.add(token.toLowerCase());
    }
  };
  const addArrayTokens = (arr: string[]) => {
    for (const value of arr ?? []) addTokens(value);
  };

  addTokens(plan.normalized_query);
  addArrayTokens(plan.hard_terms);
  addArrayTokens(plan.soft_terms);
  addArrayTokens(plan.expansions);
  addArrayTokens(plan.entities);
  addArrayTokens(plan.acronyms);

  return tokens;
}

/**
 * Collect only must-have terms (hard_terms) from the planner output.
 * These tokens gate the proximity heuristic so we treat them separately.
 */
export function buildHardTokenSet(plan: ChatNotesPlanResponse): Set<string> {
  const tokens = new Set<string>();
  for (const value of plan.hard_terms ?? []) {
    const { tokens: tokenList } = tokenizeForSearch(value);
    for (const token of tokenList) {
      tokens.add(token.toLowerCase());
    }
  }
  return tokens;
}

/**
 * Extract phrases we should attempt exact/anchor matching for.
 * We keep both quoted strings and multi-token expansions so the passage scorer
 * can reward windows that respect user phrasing.
 */
export function collectQuotedPhrases(plan: ChatNotesPlanResponse): string[] {
  const phrases = new Set<string>();
  const addPhrase = (value: string) => {
    if (!value) return;
    const trimmed = value.trim();
    if (!trimmed) return;
    if (trimmed.includes(' ')) phrases.add(trimmed);
    if (/["“”„«»]/.test(trimmed)) phrases.add(trimmed.replace(/["“”„«»]/g, '').trim());
  };

  addPhrase(plan.normalized_query);
  for (const value of plan.expansions ?? []) addPhrase(value);
  for (const value of plan.hard_terms ?? []) addPhrase(value);
  for (const value of plan.soft_terms ?? []) addPhrase(value);
  for (const value of plan.entities ?? []) addPhrase(value);

  return Array.from(phrases).filter(Boolean);
}

function computeEntityOverlapScore(body: string, entities: string[]): number {
  if (!entities.length) return 0;
  const hits = entities.reduce((acc, entity) => {
    if (!entity) return acc;
    return phraseMatch(body, entity) ? acc + 1 : acc;
  }, 0);
  return hits > 0 ? hits / entities.length : 0;
}

function computeTitleMatchScore(title: string, queryTokens: Set<string>): number {
  if (!title || queryTokens.size === 0) return 0;
  const { tokens } = tokenizeForSearch(title);
  if (!tokens.length) return 0;
  let matches = 0;
  for (const token of tokens) {
    if (queryTokens.has(token.toLowerCase())) {
      matches += 1;
    }
  }
  return matches / tokens.length;
}

function computeHeadingBoost(headings: string[], queryTerms: Set<string>): number {
  if (!headings.length) return 0;
  const path = headings;
  const count = headingPathMatchCount(path, queryTerms);
  if (count <= 0) return 0;
  return Math.min(1, count / 3);
}

function computeTagNotebookOverlap(
  note: NoteDetails,
  notebookPath: string | null,
  plan: ChatNotesPlanResponse,
): number {
  const filterTags = plan.filters?.tags ?? [];
  const filterNotebooks = plan.filters?.notebooks ?? [];
  if (filterTags.length === 0 && filterNotebooks.length === 0) {
    return 0;
  }

  const noteTags = dedupeStrings(note.tags ?? [], (value) => normalizeText(value));
  const filtersNormalized = dedupeStrings(filterTags, (value) => normalizeText(value));
  const notebookFilters = dedupeStrings(filterNotebooks, (value) => normalizeText(value));

  let matches = 0;
  for (const filter of filtersNormalized) {
    if (filter && noteTags.includes(filter)) {
      matches += 1;
    }
  }

  if (notebookFilters.length > 0 && notebookPath) {
    const normalizedPath = normalizeText(notebookPath);
    for (const notebookFilter of notebookFilters) {
      if (notebookFilter && normalizedPath.includes(notebookFilter)) {
        matches += 1;
        break;
      }
    }
  }

  const total = filtersNormalized.length + (notebookFilters.length > 0 ? 1 : 0);
  if (total === 0) return 0;
  return matches / total;
}

function computePhraseProximity(tokens: string[], hardTokens: Set<string>, allTokens: Set<string>): number {
  if (!tokens.length) return 0;
  return spanProximityScore(tokens, {
    hardTerms: hardTokens,
    allTerms: allTokens,
  });
}

interface NoteAnalysis {
  candidate: LexicalCandidate;
  termFrequencies: Record<string, number>;
}

/**
 * Execute multi-query lexical retrieval with RRF fusion and feature scoring.
 * The function fetches Joplin search results for each planned query, fuses them
 * deterministically, then computes the blended score0 used to seed later steps.
 */
export async function runLexicalRetrieval(
  plan: ChatNotesPlanResponse,
  queries: PlanQueries,
  config: LexicalRetrievalConfig,
  contextNotes: Set<string>,
): Promise<LexicalRetrievalResult> {
  const candidateLimit = Math.max(1, config.candidateLimit);
  const highlightRadius = config.highlightRadius ?? DEFAULT_HIGHLIGHT_RADIUS;
  const recencyWindowDays = config.recencyWindowDays ?? DEFAULT_RECENCY_WINDOW;

  const rrfInputs: {
    label: QueryLabel;
    query: string;
    hits: RawSearchHit[];
  }[] = [];

  for (const searchQuery of queries.queries) {
    if (!searchQuery.query?.trim()) continue;
    const hits = await fetchSearchResults(searchQuery.query, candidateLimit);
    rrfInputs.push({ label: searchQuery.label, query: searchQuery.query, hits });
  }

  let totalHits = rrfInputs.reduce((sum, entry) => sum + entry.hits.length, 0);

  if (totalHits === 0) {
    const fallbackPieces = [
      queries.normalizedQuery,
      queries.hardTerms?.join(' ') ?? '',
    ].map((value) => (value ?? '').trim()).filter(Boolean);
    const fallbackQuery = fallbackPieces.length ? fallbackPieces.join(' ') : '';
    if (fallbackQuery) {
      const fallbackHits = await fetchSearchResults(fallbackQuery, candidateLimit);
      if (fallbackHits.length > 0) {
        rrfInputs.push({
          label: 'fallback',
          query: fallbackQuery,
          hits: fallbackHits,
        });
        totalHits += fallbackHits.length;
      }
    }
  }

  if (rrfInputs.length === 0) {
    return {
      candidates: [],
      rrfLogs: [],
    };
  }

  if (totalHits === 0) {
    return {
      candidates: [],
      rrfLogs: rrfInputs.map((entry) => ({
        query: entry.label,
        hits: entry.hits.length,
        topIds: [],
      })),
    };
  }

  const rrfInput = rrfInputs.map((entry) => ({
    label: entry.label,
    items: entry.hits.map((hit, index) => ({
      id: hit.id,
      score: 1 / (index + 1),
    })),
  }));

  const rrfOutput = reciprocalRankFusion(rrfInput, {
    maxResults: candidateLimit,
  });

  const fusedById = new Map<string, RrfResult>();
  for (const result of rrfOutput.results) {
    fusedById.set(result.id, result);
  }

  const noteDetails = await Promise.all(
    rrfOutput.results.map(async (result) => {
      const detail = await fetchNoteDetails(result.id);
      return detail;
    }),
  );

  const queryTokenSet = buildQueryTokenSet(plan);
  const hardTokenSet = buildHardTokenSet(plan);
  const contextNoteSet = contextNotes;
  const entityList = plan.entities ?? [];

  const analyses: NoteAnalysis[] = [];
  const docFrequencies = new Map<string, number>();
  let totalDocumentLength = 0;
  let documentCount = 0;

  for (let index = 0; index < noteDetails.length; index += 1) {
    const detail = noteDetails[index];
    if (!detail) continue;
    const fused = fusedById.get(detail.id);
    if (!fused) continue;
    documentCount += 1;

    const notebookPath = await resolveNotebookPath(detail.parent_id ?? null);
    const snippetSource = rrfInputs
      .flatMap((entry) => entry.hits.filter((hit) => hit.id === detail.id))
      .find(Boolean);
    const snippet = extractSnippet(detail, snippetSource?.highlight ?? snippetSource?.body, highlightRadius);
    const headings = [detail.title, ...extractHeadings(detail.body)];

    const tokenized = tokenizeForSearch(detail.body ?? '');
    const tokensLower = tokenized.tokens.map((token) => token.toLowerCase());

    const termFreqs: Record<string, number> = {};
    const seenInDocument = new Set<string>();
    for (const token of tokensLower) {
      if (queryTokenSet.has(token)) {
        termFreqs[token] = (termFreqs[token] ?? 0) + 1;
        seenInDocument.add(token);
      }
    }
    for (const token of seenInDocument) {
      docFrequencies.set(token, (docFrequencies.get(token) ?? 0) + 1);
    }
    totalDocumentLength += tokensLower.length;

    const updatedAtIso = toIso(detail.user_updated_time);
    const recencyScore = recencyBoost(updatedAtIso, new Date(), recencyWindowDays);
    const prevContextScore = contextNoteBoost(detail.id, contextNoteSet);
    const headingBoost = computeHeadingBoost(headings, queryTokenSet);
    const phraseScore = computePhraseProximity(tokensLower, hardTokenSet, queryTokenSet);
    const entityOverlap = computeEntityOverlapScore(detail.body ?? '', entityList);
    const tagNotebookOverlap = computeTagNotebookOverlap(detail, notebookPath, plan);
    const titleMatch = computeTitleMatchScore(detail.title ?? '', queryTokenSet);

    const candidate: LexicalCandidate = {
      id: detail.id,
      title: detail.title ?? '',
      body: detail.body ?? '',
      snippet,
      headings,
      updatedAt: updatedAtIso,
      notebookPath,
      tags: detail.tags ?? [],
      fusedScore: fused.fusedScore,
      firstRank: fused.firstRank,
      score0: 0,
      features: {
        bm25: 0,
        headingBoost,
        phraseProximity: phraseScore,
        entityOverlap,
        tagNotebookOverlap,
        recencyBoost: recencyScore,
        prevContextBoost: prevContextScore,
        titleMatch,
      },
      contributions: fused.contributions,
      tokens: tokensLower,
      normalizedBody: tokenized.normalized,
    };

    analyses.push({
      candidate,
      termFrequencies: termFreqs,
    });
  }

  if (!analyses.length) {
    return {
      candidates: [],
      rrfLogs: rrfOutput.logs,
    };
  }

  const averageDocumentLength = totalDocumentLength > 0 && documentCount > 0
    ? totalDocumentLength / documentCount
    : 1;

  const docFrequencyRecord: Record<string, number> = {};
  for (const [token, value] of docFrequencies.entries()) {
    docFrequencyRecord[token] = value;
  }

  const queryTermsArray = Array.from(queryTokenSet);

  for (const analysis of analyses) {
    const candidate = analysis.candidate;
    const bm25 = computeBm25LScore({
      termFrequencies: analysis.termFrequencies,
      documentLength: candidate.tokens.length,
      averageDocumentLength,
      totalDocuments: documentCount,
      documentFrequencies: docFrequencyRecord,
      queryTerms: queryTermsArray,
    });
    candidate.features.bm25 = bm25;
    candidate.score0 =
      (1.0 * candidate.features.bm25) +
      (0.7 * candidate.features.headingBoost) +
      (0.5 * candidate.features.phraseProximity) +
      (0.4 * candidate.features.entityOverlap) +
      (0.2 * candidate.features.tagNotebookOverlap) +
      (0.15 * candidate.features.recencyBoost) +
      (0.1 * candidate.features.prevContextBoost) +
      (0.1 * candidate.features.titleMatch);
  }

  analyses.sort((a, b) => {
    if (b.candidate.score0 !== a.candidate.score0) {
      return b.candidate.score0 - a.candidate.score0;
    }
    if (b.candidate.fusedScore !== a.candidate.fusedScore) {
      return b.candidate.fusedScore - a.candidate.fusedScore;
    }
    return a.candidate.id.localeCompare(b.candidate.id);
  });

  const finalCandidates = analyses.map((analysis) => analysis.candidate);

  console.info('Jarvis lexical retrieval RRF logs', rrfOutput.logs);

  return {
    candidates: finalCandidates,
    rrfLogs: rrfOutput.logs,
  };
}
