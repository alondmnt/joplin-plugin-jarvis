import { ChatNotesPlanPromptInput, ChatNotesPlanResponse, ChatNotesPlanTurn, buildChatNotesPlanPrompt, validateChatNotesPlanResponse } from '../prompts/chatWithNotes';
import { LruCache } from './cache';
import { buildFallbackPlanFromText } from './planFallback';

export interface PlannerInput {
  scopeSummary: string;
  citedNotes: Array<{ id: string; title: string }>;
  conversation: ChatNotesPlanTurn[];
}

export interface PlannerOptions {
  signal?: AbortSignal;
  disableCache?: boolean;
}

export interface PlannerResult {
  plan: ChatNotesPlanResponse;
  rawResponse: string;
  fromCache: boolean;
  repaired: boolean;
  usedFallback: boolean;
}

type PlannerModel = {
  chat(prompt: string, preview?: boolean, abortSignal?: AbortSignal): Promise<string>;
};

type CachedEntry = {
  result: PlannerResult;
  timestamp: number;
};

const planCache = new LruCache<string, CachedEntry>({ maxSize: 32, entryTtlMs: 30 * 60 * 1000 });

function buildCacheKey(input: PlannerInput): string {
  const payload = {
    scope: input.scopeSummary ?? '',
    cited: input.citedNotes?.map((note) => ({ id: note.id, title: note.title })) ?? [],
    conversation: input.conversation?.map((turn) => ({ role: turn.role, content: turn.content })) ?? [],
  };
  return JSON.stringify(payload);
}

function getLastUserMessage(conversation: ChatNotesPlanTurn[]): string {
  for (let i = conversation.length - 1; i >= 0; i -= 1) {
    const turn = conversation[i];
    if (turn.role === 'user' && turn.content?.trim()) {
      return turn.content.trim();
    }
  }
  return conversation.length ? conversation[conversation.length - 1].content ?? '' : '';
}

function extractJsonObject(raw: string): string {
  if (!raw) {
    throw new Error('Planner produced empty response');
  }
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Planner response did not contain a JSON object');
  }
  return raw.slice(start, end + 1);
}

function buildPromptInput(input: PlannerInput): ChatNotesPlanPromptInput {
  return {
    scopeSummary: input.scopeSummary ?? '',
    citedNotes: input.citedNotes ?? [],
    conversation: input.conversation ?? [],
  };
}

/**
 * Reset the in-memory planner cache (primarily for tests or manual invalidation).
 */
export function clearPlannerCache() {
  planCache.clear();
}

/**
 * Run the planner LLM (with caching + repair + fallback) to obtain the retrieval plan.
 * Returns structured metadata plus flags indicating cache hits or fallback usage.
 */
export async function generatePlan(
  model: PlannerModel,
  input: PlannerInput,
  options: PlannerOptions = {},
): Promise<PlannerResult> {
  const cacheKey = buildCacheKey(input);
  if (!options.disableCache) {
    const cached = planCache.get(cacheKey);
    if (cached) {
      return { ...cached.result, fromCache: true };
    }
  }

  const promptInput = buildPromptInput(input);
  const basePrompt = buildChatNotesPlanPrompt(promptInput);

  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const repairSuffix = attempt === 0
      ? ''
      : `
The previous response could not be parsed (${String(lastError).slice(0, 120)}). Respond again with VALID JSON only.`;
    const prompt = `${basePrompt}${repairSuffix}`;
    const raw = await model.chat(prompt, false, options.signal);

    try {
      const jsonPayload = extractJsonObject(raw);
      const parsed = JSON.parse(jsonPayload);
      const valid = validateChatNotesPlanResponse(parsed);
      const result: PlannerResult = {
        plan: valid,
        rawResponse: raw,
        fromCache: false,
        repaired: attempt === 1,
        usedFallback: false,
      };
      if (!options.disableCache) {
        planCache.set(cacheKey, { result, timestamp: Date.now() });
      }
      return result;
    } catch (error) {
      lastError = error;
    }
  }

  const fallbackSource = getLastUserMessage(input.conversation ?? []);
  const fallbackPlan = buildFallbackPlanFromText(fallbackSource || input.scopeSummary || '');
  const fallbackResult: PlannerResult = {
    plan: fallbackPlan,
    rawResponse: '',
    fromCache: false,
    repaired: false,
    usedFallback: true,
  };
  if (!options.disableCache) {
    planCache.set(cacheKey, { result: fallbackResult, timestamp: Date.now() });
  }
  return fallbackResult;
}
