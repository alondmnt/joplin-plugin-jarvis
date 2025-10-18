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
import { detectDominantScope, applyScopeToQuery, type CandidateMeta } from './dominantScope';

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
  prf?: {
    analysedNotes: number;
    terms: string[];
    hits: number;
  };
  dominantScope?: {
    field: 'notebook' | 'tag';
    value: string;
    share: number;
    count: number;
  };
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
const PRF_NOTE_LIMIT = 8;
const PRF_TERM_LIMIT = 8;

function toIso(ms: number | undefined | null): string | null {
  if (!ms && ms !== 0) return null;
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString();
}

/** Deduplicate arbitrary strings with optional normalization, preserving original order. */
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

/** Prepare a token for use in a Joplin query; quotes phrases or fielded terms as needed. */
function escapeTermForQuery(term: string): string {
  const trimmed = term?.trim() ?? '';
  if (!trimmed) return '';
  const escaped = trimmed.replace(/"/g, '\\"');
  return (/\s/.test(trimmed) || /[:()]/.test(trimmed)) ? `"${escaped}"` : escaped;
}

/** Merge filters with a query string, optionally placing filters first to keep `any:1` blocks ANDed. */
function appendFiltersToQuery(q: string | null, filters: string, preferFiltersFirst = false): string | null {
  if (!q || !q.trim()) return null;
  if (!filters) return q.trim();
  return preferFiltersFirst ? `${filters} ${q.trim()}` : `${q.trim()} ${filters}`;
}

/** Hit Joplin's search API with the provided query, paging until we reach the candidate limit. */
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

/** Pull note metadata/body for scoring; returns null if the note cannot be retrieved. */
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

/**
 * Derive pseudo-relevance feedback terms from the top base hits.
 * Only tokens not already in the planner's query are considered; heading-occurring terms receive extra weight.
 */
async function computePseudoRelevanceFeedback(
  baseHits: RawSearchHit[],
  plan: ChatNotesPlanResponse,
  getNoteDetails: (noteId: string) => Promise<NoteDetails | null>,
): Promise<{ terms: string[]; analysedNotes: number }> {
  if (!baseHits.length) {
    return { terms: [], analysedNotes: 0 };
  }
  const topIds: string[] = [];
  for (const hit of baseHits) {
    if (!hit?.id) continue;
    if (!topIds.includes(hit.id)) {
      topIds.push(hit.id);
    }
    if (topIds.length >= PRF_NOTE_LIMIT) break;
  }
  if (!topIds.length) {
    return { terms: [], analysedNotes: 0 };
  }

  const details = await Promise.all(topIds.map((id) => getNoteDetails(id)));
  const validDetails = details.filter((detail): detail is NoteDetails => Boolean(detail && detail.body));
  if (!validDetails.length) {
    return { terms: [], analysedNotes: 0 };
  }

  const queryTokens = buildQueryTokenSet(plan);
  const hardTokens = buildHardTokenSet(plan);
  const bannedTokens = new Set<string>([...queryTokens, ...hardTokens]);

  const addTokensFromArray = (values: string[] | undefined) => {
    if (!values) return;
    for (const value of values) {
      const tokens = tokenizeForSearch(value).tokens;
      for (const token of tokens) {
        bannedTokens.add(token.toLowerCase());
      }
    }
  };

  addTokensFromArray(plan.expansions);
  addTokensFromArray(plan.soft_terms);
  addTokensFromArray(plan.entities);
  addTokensFromArray(plan.acronyms);

  const termTf = new Map<string, number>();
  const termDf = new Map<string, number>();
  const headingTerms = new Set<string>();

  for (const detail of validDetails) {
    const tokensLower = tokenizeForSearch(detail.body ?? '').tokens.map((token) => token.toLowerCase());
    const seen = new Set<string>();
    for (const token of tokensLower) {
      if (bannedTokens.has(token)) continue;
      if (token.length < 3) continue;
      if (/^\d+$/.test(token)) continue;
      termTf.set(token, (termTf.get(token) ?? 0) + 1);
      seen.add(token);
    }
    for (const token of seen) {
      termDf.set(token, (termDf.get(token) ?? 0) + 1);
    }
    const headings = extractHeadings(detail.body ?? '');
    for (const heading of headings) {
      const headingTokens = tokenizeForSearch(heading).tokens.map((token) => token.toLowerCase());
      for (const token of headingTokens) {
        if (token.length < 3) continue;
        headingTerms.add(token);
      }
    }
  }

  const analysedNotes = validDetails.length;
  const scoredTerms: Array<{ term: string; score: number }> = [];
  const totalDocuments = analysedNotes;

  for (const [term, tf] of termTf.entries()) {
    const df = termDf.get(term) ?? 0;
    const idf = Math.log((totalDocuments + 1) / (df + 0.5));
    if (idf < 1.0) continue;
    let score = idf * tf;
    if (headingTerms.has(term)) {
      score *= 1.2;
    }
    scoredTerms.push({ term, score });
  }

  scoredTerms.sort((a, b) => b.score - a.score);
  const selected = scoredTerms.slice(0, PRF_TERM_LIMIT).map((item) => item.term);
  return { terms: selected, analysedNotes };
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

  const noteDetailCache = new Map<string, NoteDetails | null>();
  const getNoteDetails = async (noteId: string): Promise<NoteDetails | null> => {
    if (!noteId) return null;
    if (noteDetailCache.has(noteId)) {
      return noteDetailCache.get(noteId) ?? null;
    }
    const detail = await fetchNoteDetails(noteId);
    noteDetailCache.set(noteId, detail);
    return detail;
  };

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
  const baseEntry = rrfInputs.find((entry) => entry.label === 'normalized');
  let prfStats = { analysedNotes: 0, terms: [] as string[], hits: 0 };
  let dominantScopeInfo: { field: 'notebook' | 'tag'; value: string; share: number; count: number } | null = null;

  if (baseEntry && baseEntry.hits.length > 0) {
    const prfResult = await computePseudoRelevanceFeedback(baseEntry.hits, plan, getNoteDetails);
    prfStats = { analysedNotes: prfResult.analysedNotes, terms: prfResult.terms, hits: 0 };
    if (prfResult.terms.length > 0) {
      const prfTerms = prfResult.terms.map(escapeTermForQuery).filter(Boolean);
      if (prfTerms.length > 0) {
        const prfQuery = appendFiltersToQuery(`any:1 ${prfTerms.join(' ')}`, queries.filters, true);
        if (prfQuery) {
          const prfHits = await fetchSearchResults(prfQuery, candidateLimit);
          prfStats.hits = prfHits.length;
          if (prfHits.length > 0) {
            rrfInputs.push({
              label: 'prf',
              query: prfQuery,
              hits: prfHits,
            });
            totalHits += prfHits.length;
          }
        }
      }
    }
  }

  if (baseEntry && baseEntry.hits.length > 0) {
    const scopeCandidates: CandidateMeta[] = [];
    const topScopeHits = baseEntry.hits.slice(0, 12);
    for (const hit of topScopeHits) {
      const detail = await getNoteDetails(hit.id);
      if (!detail) continue;
      const notebookPath = await resolveNotebookPath(detail.parent_id ?? null);
      scopeCandidates.push({
        noteId: detail.id,
        notebook: notebookPath,
        tags: detail.tags,
      });
      if (scopeCandidates.length >= 10) {
        break;
      }
    }
    const scope = detectDominantScope(scopeCandidates, { topN: 10, minShare: 0.5, minCount: 3, mode: 'auto' });
    if (scope) {
      const expansionsEntry = rrfInputs.find((entry) => entry.label === 'expansions');
      const sourceQuery = expansionsEntry?.query ?? baseEntry.query;
      const scopedQuery = applyScopeToQuery(sourceQuery, scope);
      if (scopedQuery && !rrfInputs.some((entry) => entry.query === scopedQuery)) {
        const scopedHits = await fetchSearchResults(scopedQuery, candidateLimit);
        if (scopedHits.length > 0) {
          rrfInputs.push({
            label: 'scoped',
            query: scopedQuery,
            hits: scopedHits,
          });
          dominantScopeInfo = {
            field: scope.field,
            value: scope.value,
            share: scope.share,
            count: scope.count,
          };
          totalHits += scopedHits.length;
        }
      }
    }
  }

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
      prf: prfStats,
      dominantScope: dominantScopeInfo ?? undefined,
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
      prf: prfStats,
      dominantScope: dominantScopeInfo ?? undefined,
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
    rrfOutput.results.map(async (result) => getNoteDetails(result.id)),
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
      prf: prfStats,
      dominantScope: dominantScopeInfo ?? undefined,
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
    prf: prfStats,
    dominantScope: dominantScopeInfo ?? undefined,
  };
}
