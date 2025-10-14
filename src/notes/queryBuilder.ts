import { ChatNotesPlanResponse } from '../prompts/chatWithNotes';

export type QueryLabel = 'normalized' | 'expansions' | 'acronyms' | 'entities' | 'title';

export interface SearchQuery { label: QueryLabel; query: string; }
export interface PlanQueries { queries: SearchQuery[]; filters: string; }

// ---- helpers ----

const toYmd = (iso: string) => iso.slice(0, 10).replace(/-/g, ''); // YYYYMMDD

const uniq = (arr: string[] = []) =>
  Array.from(new Set(arr.map(s => s?.trim()).filter(Boolean))) as string[];

function escapeTerm(term: string): string {
  const t = term?.trim() ?? '';
  if (!t) return '';
  const escaped = t.replace(/"/g, '\\"');
  // Quote phrases or anything with spaces/field punctuation
  return (/\s/.test(t) || /[:()]/.test(t)) ? `"${escaped}"` : escaped;
}

function composeFilterStrings(plan: ChatNotesPlanResponse): string {
  const clauses: string[] = [];

  for (const tag of uniq(plan.filters.tags)) {
    const e = escapeTerm(tag);
    if (e) clauses.push(`tag:${e}`);
  }
  for (const nb of uniq(plan.filters.notebooks)) {
    const e = escapeTerm(nb);
    if (e) clauses.push(`notebook:${e}`);
  }

  const { created_after: ca, created_before: cb } = plan.filters ?? {};
  // Joplin range: created:YYYYMMDD (on/after), -created:YYYYMMDD (exclude on/after => before)
  if (ca) clauses.push(`created:${toYmd(ca)}`);
  if (cb) clauses.push(`-created:${toYmd(cb)}`);

  return clauses.join(' ');
}

function joinAnd(terms: string[]): string {
  const items = uniq(terms).map(escapeTerm).filter(Boolean);
  return items.join(' ');
}

/**
 * Build an OR block. If there's only one item, return it directly (no any:1).
 */
function joinOrAny1(terms: string[]): string | null {
  const items = uniq(terms).map(escapeTerm).filter(Boolean);
  if (!items.length) return null;
  if (items.length === 1) return items[0];
  return `any:1 ${items.join(' ')}`;
}

/**
 * Append filters. For OR-queries created by any:1, we must put filters BEFORE the any:1 block
 * so filters stay AND'd and aren't accidentally OR'ed.
 */
function appendFilters(q: string | null, filters: string, preferFiltersFirst = false): string | null {
  if (!q || !q.trim()) return null;
  if (!filters) return q.trim();
  return preferFiltersFirst ? `${filters} ${q.trim()}` : `${q.trim()} ${filters}`;
}

// ---- main ----

function capSet(values: string[] | undefined, limit: number): string[] {
  if (!values || values.length === 0) return [];
  return Array.from(new Set(values.map((value) => value?.trim()).filter(Boolean))).slice(0, limit);
}

function ensureHardTerms(plan: ChatNotesPlanResponse): string[] {
  const capped = capSet(plan.hard_terms, 4);
  if (capped.length > 0) {
    return capped;
  }
  const fallback = plan.normalized_query
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  if (fallback.length > 0) {
    return [fallback[0]];
  }
  return ['notes'];
}

function sanitizePlan(plan: ChatNotesPlanResponse): ChatNotesPlanResponse {
  return {
    ...plan,
    expansions: capSet(plan.expansions, 6),
    soft_terms: capSet(plan.soft_terms, 6),
    acronyms: capSet(plan.acronyms, 6),
    entities: capSet(plan.entities, 5),
    hard_terms: ensureHardTerms(plan),
  };
}

/**
 * Translate the planner response into concrete Joplin search queries.
 * Handles quoting, OR blocks (`any:1`), and filter composition for Step B.
 */
export function buildQueriesFromPlan(plan: ChatNotesPlanResponse): PlanQueries {
  const cleaned = sanitizePlan(plan);
  const filters = composeFilterStrings(cleaned);
  const queries: SearchQuery[] = [];

  // Q1: Base AND (normalized + hard_terms)
  const baseParts: string[] = [];
  if (cleaned.normalized_query?.trim()) baseParts.push(cleaned.normalized_query.trim());
  if (cleaned.hard_terms?.length)       baseParts.push(joinAnd(cleaned.hard_terms));
  const baseQuery = appendFilters(baseParts.join(' ').trim(), filters);
  if (baseQuery) queries.push({ label: 'normalized', query: baseQuery });

  // Q2: Expansions OR (expansions ∪ soft_terms) — acronyms get their own query later
  const expansionPool = [
    ...(cleaned.expansions ?? []),
    ...(cleaned.soft_terms ?? []),
  ];
  const exOr = joinOrAny1(expansionPool);
  const exQuery = appendFilters(exOr, filters, true); // filters BEFORE any:1
  if (exQuery) queries.push({ label: 'expansions', query: exQuery });

  // Q3: Title focus (normalized + hard_terms mapped to title:)
  const titleTerms: string[] = [];
  if (cleaned.normalized_query?.trim()) titleTerms.push(cleaned.normalized_query.trim());
  if (cleaned.hard_terms?.length)       titleTerms.push(...cleaned.hard_terms);
  const titleBits = uniq(titleTerms).map(t => `title:${escapeTerm(t)}`).join(' ');
  const titleQuery = appendFilters(titleBits || null, filters);
  if (titleQuery) queries.push({ label: 'title', query: titleQuery });

  // Q4: Acronyms OR (dedicated narrower query)
  const acOr = joinOrAny1(cleaned.acronyms ?? []);
  const acQuery = appendFilters(acOr, filters, true); // filters BEFORE any:1
  if (acQuery) queries.push({ label: 'acronyms', query: acQuery });

  // Q5: Entities OR
  const entOr = joinOrAny1(cleaned.entities ?? []);
  const entQuery = appendFilters(entOr, filters, true); // filters BEFORE any:1
  if (entQuery) queries.push({ label: 'entities', query: entQuery });

  return { queries, filters };
}
