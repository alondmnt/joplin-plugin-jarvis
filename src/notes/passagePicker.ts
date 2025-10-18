import type { ChatNotesPlanResponse } from '../prompts/chatWithNotes';
import { tokenizeForSearch } from './tokenizer';
import {
  computeBm25LScore,
  spanProximityScore,
  anchorQuoteMatchCount,
  headingPathMatchCount,
  contextNoteBoost,
  selectWithMmr,
} from './passageScorer';
import { splitTextIntoTokenWindows, type WindowSizerOptions } from './windowSizer';
import type { LexicalCandidate } from './lexicalRetrieval';
import {
  buildHardTokenSet,
  buildQueryTokenSet,
  collectQuotedPhrases,
} from './lexicalRetrieval';

export interface PassageFeatureScores {
  bm25: number;
  spanProximity: number;
  anchorBoost: number;
  headingBoost: number;
  contextBoost: number;
  recencyBoost: number;
}

export interface PassageSelection {
  noteId: string;
  noteTitle: string;
  headingPath: string[];
  excerpt: string;
  score: number;
  noteScore0: number;
  features: PassageFeatureScores;
  updatedAt: string | null;
}

export interface PassagePickerOptions {
  maxPassagesPerNote: number;
  windowTokens: number;
  strideRatio?: number;
  mmrLambda?: number;
  maxOverlapRatio?: number;
  longNoteWindowCap?: number;
  longNoteTokenCap?: number;
}

interface Section {
  headingPath: string[];
  text: string;
}

interface WindowCandidate {
  id: string;
  excerpt: string;
  headingPath: string[];
  start: number;
  end: number;
  tokens: string[];
  termFrequencies: Record<string, number>;
  score: number;
  features: PassageFeatureScores;
}

const DEFAULT_WINDOW_TOKENS = 150;
const DEFAULT_STRIDE = 0.5;
const DEFAULT_MMR_LAMBDA = 0.7;
const DEFAULT_MAX_OVERLAP = 0.6;
const DEFAULT_LONG_NOTE_WINDOW_CAP = 80;
const DEFAULT_LONG_NOTE_TOKEN_CAP = 18000;
const MAX_PASSAGE_EXCERPT_CHARS = 400;
const TEMPORAL_AGGREGATE_REGEX = /\b(total|totals|sum|sums|duration|durations|hours?|minutes?|time|times|spent|logged|logging|track(?:ing)?)\b/;
const TEMPORAL_CO_OCCURRENCE_RADIUS = 8;
const AGGREGATE_KEYWORDS = new Set([
  'total',
  'totals',
  'sum',
  'sums',
  'duration',
  'durations',
  'hour',
  'hours',
  'minute',
  'minutes',
  'time',
  'times',
  'spent',
  'spend',
  'logged',
  'logging',
  'log',
  'track',
  'tracking',
]);
const MONTH_NAME_TOKENS = new Set([
  'january', 'jan',
  'february', 'feb',
  'march', 'mar',
  'april', 'apr',
  'may',
  'june', 'jun',
  'july', 'jul',
  'august', 'aug',
  'september', 'sep', 'sept',
  'october', 'oct',
  'november', 'nov',
  'december', 'dec',
]);

interface TemporalGate {
  required: boolean;
  monthTokens: string[];
  taskTokens: Set<string>;
}

function isMonthToken(token: string): boolean {
  if (!token) return false;
  const lower = token.toLowerCase();
  if (MONTH_NAME_TOKENS.has(lower)) return true;
  if (/^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[-_]?\d{4}$/.test(lower)) return true;
  if (/^\d{4}[-/](?:0[1-9]|1[0-2])$/.test(lower)) return true;
  if (/^(?:0?[1-9]|1[0-2])[-/]\d{4}$/.test(lower)) return true;
  if (/^\d{4}-\d{2}-\d{2}$/.test(lower)) return true;
  if (/^\d{4}\/\d{2}\/\d{2}$/.test(lower)) return true;
  return false;
}

function extractMonthFragment(value: string | null | undefined): string | null {
  if (!value) return null;
  if (value.length >= 7) {
    return value.slice(0, 7).toLowerCase();
  }
  return null;
}

function buildTemporalGate(plan: ChatNotesPlanResponse, queryTokens: Set<string>): TemporalGate {
  const monthTokens = new Set<string>();
  for (const token of queryTokens) {
    if (isMonthToken(token)) {
      monthTokens.add(token.toLowerCase());
    }
  }

  const addDateToken = (value: string | null | undefined) => {
    const fragment = extractMonthFragment(value);
    if (fragment) {
      monthTokens.add(fragment);
    }
  };

  const filters = plan.filters ?? {
    created_after: null,
    created_before: null,
    updated_after: null,
    updated_before: null,
  };

  addDateToken(filters.updated_after);
  addDateToken(filters.updated_before);
  addDateToken(filters.created_after);
  addDateToken(filters.created_before);

  const textSources = [
    plan.normalized_query,
    ...(plan.expansions ?? []),
    ...(plan.hard_terms ?? []),
    ...(plan.soft_terms ?? []),
  ];
  const combinedText = textSources.filter(Boolean).join(' ').toLowerCase();
  const aggregateIntent = TEMPORAL_AGGREGATE_REGEX.test(combinedText);
  const hasTemporalFilter = Boolean(
    filters.updated_after ||
    filters.updated_before ||
    filters.created_after ||
    filters.created_before,
  );
  const hasMonthIntent = monthTokens.size > 0 || hasTemporalFilter;

  const taskTokens = new Set<string>();
  const addTaskTokens = (value: string | null | undefined) => {
    if (!value) return;
    const tokens = tokenizeForSearch(value).tokens;
    for (const token of tokens) {
      const lower = token.toLowerCase();
      if (!lower) continue;
      if (isMonthToken(lower)) continue;
      if (AGGREGATE_KEYWORDS.has(lower)) continue;
      taskTokens.add(lower);
    }
  };

  for (const term of plan.hard_terms ?? []) addTaskTokens(term);
  if (taskTokens.size === 0) {
    addTaskTokens(plan.normalized_query);
  }

  const required = aggregateIntent && hasMonthIntent && taskTokens.size > 0;

  return {
    required,
    monthTokens: Array.from(monthTokens),
    taskTokens,
  };
}

function matchesMonthToken(token: string, gate: TemporalGate): boolean {
  if (!token) return false;
  const lower = token.toLowerCase();
  if (isMonthToken(lower)) return true;
  for (const monthToken of gate.monthTokens) {
    if (monthToken && lower.includes(monthToken)) {
      return true;
    }
  }
  return false;
}

function windowPassesTemporalGate(tokens: string[], gate: TemporalGate): boolean {
  if (!gate.required) return true;
  if (!tokens.length) return false;
  const monthIndexes: number[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (matchesMonthToken(token, gate)) {
      monthIndexes.push(index);
    }
  }
  if (monthIndexes.length === 0) return false;
  for (const monthIndex of monthIndexes) {
    const start = Math.max(0, monthIndex - TEMPORAL_CO_OCCURRENCE_RADIUS);
    const end = Math.min(tokens.length - 1, monthIndex + TEMPORAL_CO_OCCURRENCE_RADIUS);
    for (let idx = start; idx <= end; idx += 1) {
      if (gate.taskTokens.has(tokens[idx])) {
        return true;
      }
    }
  }
  return false;
}

function splitIntoSections(note: LexicalCandidate): Section[] {
  const sections: Section[] = [];
  const lines = (note.body ?? '').split('\n');
  const headingLevels: string[] = [];
  let buffer: string[] = [];

  const pushSection = () => {
    if (buffer.length === 0) return;
    const text = buffer.join('\n').trim();
    buffer = [];
    if (!text) return;
    const headingPath = [note.title, ...headingLevels.filter(Boolean)];
    sections.push({ headingPath, text });
  };

  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.*)$/);
    if (match) {
      pushSection();
      const level = match[1].length;
      const heading = match[2].trim();
      headingLevels.splice(level - 1);
      headingLevels[level - 1] = heading;
    } else {
      buffer.push(line);
    }
  }
  pushSection();

  if (sections.length === 0 && note.body?.trim()) {
    sections.push({
      headingPath: [note.title],
      text: note.body.trim(),
    });
  }

  return sections;
}

function buildWindowCandidates(
  note: LexicalCandidate,
  contextNotes: Set<string>,
  options: PassagePickerOptions,
  queryTokens: Set<string>,
  hardTokens: Set<string>,
  quotedPhrases: string[],
): WindowCandidate[] {
  const sections = splitIntoSections(note);
  if (sections.length === 0) return [];

  const windows: WindowCandidate[] = [];
  let noteTokenOffset = 0;

  const windowOptions: WindowSizerOptions = {
    maxTokens: options.windowTokens || DEFAULT_WINDOW_TOKENS,
    strideRatio: options.strideRatio ?? DEFAULT_STRIDE,
  };

  for (let sectionIndex = 0; sectionIndex < sections.length; sectionIndex += 1) {
    const section = sections[sectionIndex];
    const tokenWindows = splitTextIntoTokenWindows(section.text, windowOptions);
    const sectionTokenCount = tokenWindows.length > 0 ? tokenWindows[tokenWindows.length - 1].end : 0;

    for (let windowIndex = 0; windowIndex < tokenWindows.length; windowIndex += 1) {
      const tokenWindow = tokenWindows[windowIndex];
      const excerpt = tokenWindow.text.trim();
      if (!excerpt) continue;
      const tokenResult = tokenizeForSearch(excerpt);
      const tokens = tokenResult.tokens.map((token) => token.toLowerCase());
      if (tokens.length === 0) continue;

      const termFrequencies: Record<string, number> = {};
      for (const token of tokens) {
        if (!queryTokens.has(token)) continue;
        termFrequencies[token] = (termFrequencies[token] ?? 0) + 1;
      }

      const id = `${note.id}#${sectionIndex}:${windowIndex}`;
      const headingPath = section.headingPath.length > 0 ? section.headingPath : [note.title];
      const anchorMatches = anchorQuoteMatchCount(excerpt, quotedPhrases);
      const headingMatch = headingPathMatchCount(headingPath, queryTokens);
      const contextBoost = contextNoteBoost(note.id, contextNotes);
      const recencyBoost = note.features.recencyBoost ?? 0;
      const spanScore = spanProximityScore(tokens, {
        hardTerms: hardTokens,
        allTerms: queryTokens,
      });

      // Feature placeholders; BM25 computed later when we have document frequencies.
      const features: PassageFeatureScores = {
        bm25: 0,
        spanProximity: spanScore,
        anchorBoost: Math.min(anchorMatches, 3) / 3,
        headingBoost: Math.min(headingMatch, 3) / 3,
        contextBoost,
        recencyBoost,
      };

      windows.push({
        id,
        excerpt,
        headingPath,
        start: noteTokenOffset + tokenWindow.start,
        end: noteTokenOffset + tokenWindow.end,
        tokens,
        termFrequencies,
        score: 0,
        features,
      });
    }

    noteTokenOffset += sectionTokenCount;
  }

  return windows;
}

function computeWindowScores(
  windows: WindowCandidate[],
  queryTokens: Set<string>,
): void {
  if (windows.length === 0) return;

  const docFreq = new Map<string, number>();
  let totalLength = 0;

  for (const window of windows) {
    const seen = new Set<string>();
    totalLength += window.tokens.length;
    for (const token of window.tokens) {
      if (!queryTokens.has(token)) continue;
      if (!seen.has(token)) {
        seen.add(token);
        docFreq.set(token, (docFreq.get(token) ?? 0) + 1);
      }
    }
  }

  const totalDocuments = windows.length;
  const averageDocumentLength = totalLength > 0 ? totalLength / totalDocuments : 1;
  const docFreqRecord: Record<string, number> = {};
  for (const [token, value] of docFreq.entries()) {
    docFreqRecord[token] = value;
  }

  const queryTermsArray = Array.from(queryTokens);

  for (const window of windows) {
    const bm25 = computeBm25LScore({
      termFrequencies: window.termFrequencies,
      documentLength: window.tokens.length,
      averageDocumentLength,
      totalDocuments,
      documentFrequencies: docFreqRecord,
      queryTerms: queryTermsArray,
    });
    window.features.bm25 = bm25;
    window.score =
      (1.0 * window.features.bm25) +
      (0.6 * window.features.spanProximity) +
      (0.5 * window.features.headingBoost) +
      (0.3 * window.features.recencyBoost) +
      (0.2 * window.features.contextBoost) +
      (0.15 * window.features.anchorBoost);
  }
}

/**
 * Score and select passages from the reranked note set using BM25L + heuristics.
 * Returns MMR-pruned windows annotated with feature scores for downstream use.
 */
export function pickPassages(
  notes: LexicalCandidate[],
  plan: ChatNotesPlanResponse,
  contextNotes: Set<string>,
  options: PassagePickerOptions,
): PassageSelection[] {
  const config: PassagePickerOptions = {
    maxPassagesPerNote: options.maxPassagesPerNote,
    windowTokens: options.windowTokens || DEFAULT_WINDOW_TOKENS,
    strideRatio: options.strideRatio ?? DEFAULT_STRIDE,
    mmrLambda: options.mmrLambda ?? DEFAULT_MMR_LAMBDA,
    maxOverlapRatio: options.maxOverlapRatio ?? DEFAULT_MAX_OVERLAP,
    longNoteWindowCap: options.longNoteWindowCap ?? DEFAULT_LONG_NOTE_WINDOW_CAP,
    longNoteTokenCap: options.longNoteTokenCap ?? DEFAULT_LONG_NOTE_TOKEN_CAP,
  };

  const selections: PassageSelection[] = [];
  const queryTokens = buildQueryTokenSet(plan);
  const hardTokens = buildHardTokenSet(plan);
  const quotedPhrases = collectQuotedPhrases(plan);
  const temporalGate = buildTemporalGate(plan, queryTokens);

  for (const note of notes) {
    const windows = buildWindowCandidates(
      note,
      contextNotes,
      config,
      queryTokens,
      hardTokens,
      quotedPhrases,
    );
    if (windows.length === 0) continue;

    const filteredWindows = windows.filter((window) => windowPassesTemporalGate(window.tokens, temporalGate));
    if (filteredWindows.length === 0) {
      continue;
    }

    const guardrailTriggered =
      windows.length > config.longNoteWindowCap || note.tokens.length > config.longNoteTokenCap;

    if (guardrailTriggered) {
      console.warn(
        'Jarvis passage picker long-note guardrail triggered',
        note.id,
        { windows: windows.length, tokens: note.tokens.length },
      );
    }

    computeWindowScores(filteredWindows, queryTokens);

    const candidatePool = guardrailTriggered
      ? [...filteredWindows]
          .sort((a, b) => b.score - a.score)
          .slice(0, Math.max(config.maxPassagesPerNote * 3, config.maxPassagesPerNote))
      : filteredWindows;

    const mmrCandidates = candidatePool.map((window) => ({
      id: window.id,
      score: window.score,
      window: { start: window.start, end: window.end },
      payload: window,
    }));

    const selected = selectWithMmr(mmrCandidates, {
      lambda: config.mmrLambda,
      maxSelections: config.maxPassagesPerNote,
      maxOverlapRatio: config.maxOverlapRatio,
    });

    for (const selection of selected) {
      const payload = selection.payload;
      const excerpt =
        payload.excerpt.length > MAX_PASSAGE_EXCERPT_CHARS
          ? `${payload.excerpt.slice(0, MAX_PASSAGE_EXCERPT_CHARS)}…`
          : payload.excerpt;
      selections.push({
        noteId: note.id,
        noteTitle: note.title,
        headingPath: payload.headingPath,
        excerpt,
        score: payload.score,
        noteScore0: note.score0,
        features: payload.features,
        updatedAt: note.updatedAt,
      });
    }
  }

  selections.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (b.noteScore0 !== a.noteScore0) return b.noteScore0 - a.noteScore0;
    if (a.noteId !== b.noteId) return a.noteId.localeCompare(b.noteId);
    return a.headingPath.join('>').localeCompare(b.headingPath.join('>'));
  });

  return selections;
}
