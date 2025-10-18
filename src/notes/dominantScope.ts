export type ScopeField = 'notebook' | 'tag';

export interface CandidateMeta {
  noteId: string;
  notebook?: string | null;
  tags?: string[] | null;
}

export interface DominantScopeOptions {
  topN?: number;
  minShare?: number;
  minCount?: number;
  mode?: 'auto' | 'notebook' | 'tag';
}

export interface DominantScope {
  field: ScopeField;
  value: string;
  count: number;
  considered: number;
  share: number;
  filter: string;
}

function escapeFilterTerm(input: string): string {
  const trimmed = (input ?? '').trim();
  if (!trimmed) return '';
  const escaped = trimmed.replace(/"/g, '\\"');
  return /\s/.test(trimmed) ? `"${escaped}"` : escaped;
}

function argmax<K extends string>(counts: Record<K, number>): [K, number] | null {
  let bestKey: K | null = null;
  let bestValue = -1;
  for (const key in counts) {
    if (!Object.prototype.hasOwnProperty.call(counts, key)) continue;
    const value = counts[key as K]!;
    if (
      value > bestValue ||
      (value === bestValue && (bestKey === null || key < bestKey))
    ) {
      bestKey = key as K;
      bestValue = value;
    }
  }
  return bestKey ? [bestKey, bestValue] : null;
}

function detectScopeForField(
  candidates: CandidateMeta[],
  field: ScopeField,
  considerN: number,
  minShare: number,
  minCount: number,
): DominantScope | null {
  const counts: Record<string, number> = Object.create(null);
  for (const candidate of candidates) {
    if (field === 'notebook') {
      const notebook = (candidate.notebook ?? '').trim();
      if (!notebook) continue;
      counts[notebook] = (counts[notebook] ?? 0) + 1;
    } else {
      const tagValues = candidate.tags ?? [];
      const unique = new Set(
        tagValues
          .filter(Boolean)
          .map((tag) => tag?.trim())
          .filter((tag): tag is string => Boolean(tag)),
      );
      for (const tag of unique) {
        counts[tag] = (counts[tag] ?? 0) + 1;
      }
    }
  }

  const winning = argmax(counts);
  if (!winning) return null;
  const [value, count] = winning;
  const share = count / considerN;
  if (count < minCount || share < minShare) {
    return null;
  }

  return {
    field,
    value,
    count,
    considered: considerN,
    share,
    filter: field === 'notebook'
      ? `notebook:${escapeFilterTerm(value)}`
      : `tag:${escapeFilterTerm(value)}`,
  };
}

export function detectDominantScope(
  candidates: CandidateMeta[],
  options: DominantScopeOptions = {},
): DominantScope | null {
  const topN = Math.max(1, options.topN ?? 10);
  const minShare = Math.min(1, Math.max(0, options.minShare ?? 0.5));
  const minCount = Math.max(1, options.minCount ?? 3);
  const mode = options.mode ?? 'auto';

  if (!candidates.length) return null;
  const inspected = candidates.slice(0, topN);
  const considered = inspected.length;
  if (!considered) return null;

  if (mode === 'notebook') {
    return detectScopeForField(inspected, 'notebook', considered, minShare, minCount);
  }
  if (mode === 'tag') {
    return detectScopeForField(inspected, 'tag', considered, minShare, minCount);
  }

  return (
    detectScopeForField(inspected, 'notebook', considered, minShare, minCount) ??
    detectScopeForField(inspected, 'tag', considered, minShare, minCount)
  );
}

export function applyScopeToQuery(query: string | null, scope: DominantScope): string | null {
  if (!query || !query.trim()) return null;
  const trimmed = query.trim();
  if (trimmed.startsWith('any:1 ')) {
    return `${scope.filter} ${trimmed}`;
  }
  return `${trimmed} ${scope.filter}`;
}
