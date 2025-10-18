export const CHAT_NOTES_PLAN_PROMPT = `You are Jarvis, the retrieval planner for a personal knowledge base hosted in Joplin.
Use the inputs to understand what the user is asking for and produce a deterministic plan describing how to search the notes.
You must respond with valid JSON only, matching the exact schema and property order shown below.

Inputs:
- Current scope: list the notebooks, tags, or filters that limit the search. If empty, say "all notes".
- Previously cited notes: zero or more note IDs with titles.
- Conversation turns: the last user/assistant exchanges, newest last.

Planning rules:
1. Normalise the user's intent into a concise search query string.
2. Provide lexical expansions (synonyms, spelling variants, paraphrases) that can improve recall.
3. Separate must-have tokens ("hard_terms") from helpful but optional tokens ("soft_terms").
4. Add salient named entities and acronyms when they help with expansions.
5. Derive structured filters from scope (tags, notebooks, created/updated time ranges).
6. Include relevant context notes (note IDs) that should stay in scope for follow-up turns.
7. Do not invent metadata. Leave arrays empty when you have nothing reliable to add.

Return JSON with this exact structure:
{
  "prompt_version": "chat-notes-plan-2",
  "normalized_query": "string",
  "expansions": ["string"],
  "hard_terms": ["string"],
  "soft_terms": ["string"],
  "entities": ["string"],
  "acronyms": ["string"],
  "filters": {
    "tags": ["string"],
    "notebooks": ["string"],
    "created_after": "ISO-8601 or null",
    "created_before": "ISO-8601 or null"
  },
  "context_notes": ["NOTE_ID"]
}

The response must be a single JSON object, no markdown or commentary.`;

export interface ChatNotesPlanFilters {
  tags: string[];
  notebooks: string[];
  created_after: string | null;
  created_before: string | null;
  updated_after?: string | null;
  updated_before?: string | null;
}

export interface ChatNotesPlanResponse {
  prompt_version: 'chat-notes-plan-2';
  normalized_query: string;
  expansions: string[];
  hard_terms: string[];
  soft_terms: string[];
  entities: string[];
  acronyms: string[];
  filters: ChatNotesPlanFilters;
  context_notes: string[];
}

export type ChatNotesPlanTurn = {
  role: 'user' | 'assistant';
  content: string;
};

export interface ChatNotesPlanPromptInput {
  scopeSummary: string;
  citedNotes: Array<{ id: string; title: string }>;
  conversation: ChatNotesPlanTurn[];
}

export function buildChatNotesPlanPrompt(input: ChatNotesPlanPromptInput): string {
  const scopeLine = input.scopeSummary.trim().length > 0 ? input.scopeSummary.trim() : 'All notes';
  const citedSection =
    input.citedNotes.length === 0
      ? 'Previously cited notes: none.'
      : `Previously cited notes:\n${input.citedNotes
          .map((note) => `- ${note.id}: ${note.title}`)
          .join('\n')}`;
  const conversation =
    input.conversation.length === 0
      ? 'No prior conversation turns.'
      : input.conversation
          .map((turn) => `${turn.role === 'assistant' ? 'Assistant' : 'User'}: ${turn.content}`)
          .join('\n');

  return `${CHAT_NOTES_PLAN_PROMPT}

Scope: ${scopeLine}
${citedSection}

Conversation:
${conversation}
`;
}

function ensureStringArray(field: string, value: unknown): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`Expected \`${field}\` to be an array`);
  }
  value.forEach((item, index) => {
    if (typeof item !== 'string') {
      throw new Error(`Expected \`${field}[${index}]\` to be a string`);
    }
  });
  return value as string[];
}

function ensureIsoOrNull(field: string, value: unknown): string | null {
  if (value === null) return null;
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`Expected \`${field}\` to be null or a non-empty ISO-8601 string`);
  }
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Expected \`${field}\` to be a valid ISO-8601 timestamp`);
  }
  return value;
}

export function validateChatNotesPlanResponse(payload: unknown): ChatNotesPlanResponse {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error('Planner response must be an object');
  }
  const obj = payload as Record<string, unknown>;
  if (obj.prompt_version !== 'chat-notes-plan-2') {
    throw new Error('`prompt_version` must equal "chat-notes-plan-2"');
  }
  if (typeof obj.normalized_query !== 'string' || obj.normalized_query.trim().length === 0) {
    throw new Error('`normalized_query` must be a non-empty string');
  }
  const expansions = ensureStringArray('expansions', obj.expansions ?? []);
  const hardTerms = ensureStringArray('hard_terms', obj.hard_terms ?? []);
  const softTerms = ensureStringArray('soft_terms', obj.soft_terms ?? []);
  const entities = ensureStringArray('entities', obj.entities ?? []);
  const acronyms = ensureStringArray('acronyms', obj.acronyms ?? []);
  const contextNotes = ensureStringArray('context_notes', obj.context_notes ?? []);

  const filtersRaw = obj.filters;
  if (typeof filtersRaw !== 'object' || filtersRaw === null) {
    throw new Error('`filters` must be an object');
  }
  const filtersRecord = filtersRaw as Record<string, unknown>;
  const filters: ChatNotesPlanFilters = {
    tags: ensureStringArray('filters.tags', filtersRecord.tags ?? []),
    notebooks: ensureStringArray('filters.notebooks', filtersRecord.notebooks ?? []),
    created_after: ensureIsoOrNull('filters.created_after', filtersRecord.created_after ?? null),
    created_before: ensureIsoOrNull('filters.created_before', filtersRecord.created_before ?? null),
    updated_after: ensureIsoOrNull('filters.updated_after', filtersRecord.updated_after ?? null),
    updated_before: ensureIsoOrNull('filters.updated_before', filtersRecord.updated_before ?? null),
  };

  return {
    prompt_version: 'chat-notes-plan-2',
    normalized_query: obj.normalized_query as string,
    expansions,
    hard_terms: hardTerms,
    soft_terms: softTerms,
    entities,
    acronyms,
    filters,
    context_notes: contextNotes,
  };
}

export const CHAT_NOTES_PLAN_EXAMPLE: ChatNotesPlanResponse = {
  prompt_version: 'chat-notes-plan-2',
  normalized_query: 'atlas mission october 2025 milestones',
  expansions: ['mission log', 'launch preparation', 'oct 2025'],
  hard_terms: ['atlas', 'mission'],
  soft_terms: ['milestones', 'timeline'],
  entities: ['Project Atlas'],
  acronyms: [],
  filters: {
    tags: [],
    notebooks: ['projects/atlas'],
    created_after: '2025-10-01',
    created_before: '2025-10-31',
  },
  context_notes: ['atlas-mission-log'],
};

export interface ChatNotesRerankCandidate {
  id: string;
  title: string;
  snippet: string;
  score0: number;
  headings?: string[];
  updated_at?: string;
}

export interface ChatNotesRerankPromptInput {
  query: string;
  normalizedQuery: string;
  candidateA: ChatNotesRerankCandidate;
  candidateB: ChatNotesRerankCandidate;
}

export const CHAT_NOTES_RERANK_PROMPT = `You are Jarvis, ranking two candidate notes for a retrieval step.
You will receive the user query, the normalised query, and metadata for candidates A and B.
Choose the candidate that is more relevant to answering the query using only the provided text.

Return valid JSON with the shape:
{
  "prompt_version": "chat-notes-rerank-2",
  "winner": "A" | "B" | "none",
  "confident": 0 | 1,
  "reason": "optional string (<=160 characters)"
}

Guidelines:
- Pick "none" only when both candidates are clearly off-topic or equally irrelevant.
- Set confident = 1 when the winner is obviously better; otherwise 0.
- Keep the reason short (<=160 characters) and reference surface facts from the snippets.
- Do not invent details or refer to hidden context.
- If you need more space, omit low-value commentary instead of exceeding 160 characters.`;

export function buildChatNotesRerankPrompt(input: ChatNotesRerankPromptInput): string {
  const formatCandidate = (label: 'A' | 'B', candidate: ChatNotesRerankCandidate) => {
    const headings =
      candidate.headings && candidate.headings.length > 0
        ? candidate.headings.join(' > ')
        : 'N/A';
    const updated = candidate.updated_at ?? 'unknown';
    const snippet = candidate.snippet.length > 500
      ? `${candidate.snippet.slice(0, 500)}…`
      : candidate.snippet;
    return `${label}) Note ID: ${candidate.id}
Title: ${candidate.title}
Headings: ${headings}
Lexical score: ${candidate.score0.toFixed(4)}
Last updated: ${updated}
Snippet:
${snippet}`;
  };

  return `${CHAT_NOTES_RERANK_PROMPT}

Query: ${input.query}
Normalised query: ${input.normalizedQuery}

${formatCandidate('A', input.candidateA)}

${formatCandidate('B', input.candidateB)}
`;
}

export interface ChatNotesRerankResponse {
  prompt_version: 'chat-notes-rerank-2';
  winner: 'A' | 'B' | 'none';
  confident: 0 | 1;
  reason?: string;
}

export function validateChatNotesRerankResponse(payload: unknown): ChatNotesRerankResponse {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error('Reranker response must be an object');
  }
  const obj = payload as Record<string, unknown>;
  if (obj.prompt_version !== 'chat-notes-rerank-2') {
    throw new Error('`prompt_version` must equal "chat-notes-rerank-2"');
  }

  const winner = obj.winner;
  if (winner !== 'A' && winner !== 'B' && winner !== 'none') {
    throw new Error('`winner` must be "A", "B", or "none"');
  }

  const confident = obj.confident;
  if (confident !== 0 && confident !== 1) {
    throw new Error('`confident` must be 0 or 1');
  }

  if (obj.reason !== undefined && typeof obj.reason !== 'string') {
    throw new Error('`reason` must be a string when provided');
  }
  if (typeof obj.reason === 'string' && obj.reason.length > 160) {
    throw new Error('`reason` must be <=160 characters when provided');
  }

  const response: ChatNotesRerankResponse = {
    prompt_version: 'chat-notes-rerank-2',
    winner,
    confident,
  };

  if (typeof obj.reason === 'string') {
    response.reason = obj.reason;
  }

  return response;
}

export const CHAT_NOTES_RERANK_EXAMPLE: ChatNotesRerankResponse = {
  prompt_version: 'chat-notes-rerank-2',
  winner: 'A',
  confident: 1,
  reason: 'Note A directly covers Atlas mission milestones in October 2025.',
};
