import type { TextGenerationModel } from '../models/models';
import type { ChatNotesPlanResponse } from '../prompts/chatWithNotes';
import type { PassageSelection } from './passagePicker';

export interface AnswerCitation {
  label: number;
  note_id: string;
  heading_path: string[];
  excerpt: string;
}

export interface AnswerComposerOptions {
  abortSignal?: AbortSignal;
}

export interface AnswerComposerResult {
  promptVersion: string;
  answerMarkdown: string;
  citations: AnswerCitation[];
  raw: string;
}

const ANSWER_PROMPT_HEADER = `You are Jarvis, an assistant generating answers from personal notes.
Use only the provided passages. If they do not contain enough information, say so explicitly.
Return valid JSON only matching the schema:
{
  "prompt_version": "chat-notes-answer-2",
  "answer_markdown": "string",
  "citations": [
    { "label": number, "note_id": "string", "heading_path": ["string"], "excerpt": "string" }
  ]
}

Guidelines:
- The answer_markdown should be helpful, concise, and include citations like [1] referencing provided labels.
- Cite every non-trivial statement with one or more labels.
- Do not invent citations or content beyond the passages.
- If information is insufficient, set answer_markdown to a short apology and return an empty citations array.
- When passages are provided you MUST include at least one citation referencing them. Returning an empty citations array is invalid in that case.
- Never include commentary outside JSON.`;

function extractJsonObject(raw: string): string {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('Answer composer response missing JSON object');
  }
  return raw.slice(start, end + 1);
}

function validateAnswerPayload(payload: unknown, maxLabel: number): AnswerComposerResult {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error('Answer composer payload must be an object');
  }
  const obj = payload as Record<string, unknown>;
  if (obj.prompt_version !== 'chat-notes-answer-2') {
    throw new Error('prompt_version must equal "chat-notes-answer-2"');
  }
  if (typeof obj.answer_markdown !== 'string') {
    throw new Error('answer_markdown must be a string');
  }
  if (!Array.isArray(obj.citations)) {
    throw new Error('citations must be an array');
  }

  const citations: AnswerCitation[] = obj.citations.map((entry, index) => {
    if (typeof entry !== 'object' || entry === null) {
      throw new Error(`citations[${index}] must be an object`);
    }
    const record = entry as Record<string, unknown>;
    const label = record.label;
    const noteId = record.note_id;
    const headingPath = record.heading_path;
    const excerpt = record.excerpt;
    if (typeof label !== 'number' || !Number.isFinite(label)) {
      throw new Error(`citations[${index}].label must be a finite number`);
    }
    if (label < 1 || label > maxLabel) {
      throw new Error(`citations[${index}].label must be between 1 and ${maxLabel}`);
    }
    if (typeof noteId !== 'string' || !noteId.trim()) {
      throw new Error(`citations[${index}].note_id must be a non-empty string`);
    }
    if (!Array.isArray(headingPath) || headingPath.some((value) => typeof value !== 'string')) {
      throw new Error(`citations[${index}].heading_path must be an array of strings`);
    }
    if (typeof excerpt !== 'string') {
      throw new Error(`citations[${index}].excerpt must be a string`);
    }
    return {
      label: Math.trunc(label),
      note_id: noteId,
      heading_path: headingPath,
      excerpt,
    };
  });

  return {
    promptVersion: 'chat-notes-answer-2',
    answerMarkdown: obj.answer_markdown,
    citations,
    raw: JSON.stringify(payload),
  };
}

/**
 * Call the answer composer prompt with deterministic JSON output.
 * We supply labelled passages and validate citations before handing the result back.
 */
export async function composeAnswer(
  model: TextGenerationModel,
  plan: ChatNotesPlanResponse,
  userQuestion: string,
  passages: PassageSelection[],
  options: AnswerComposerOptions = {},
): Promise<AnswerComposerResult> {
  const labelledPassages = passages.map((passage, index) => ({
    label: index + 1,
    ...passage,
  }));

  const passagesSection = labelledPassages.length === 0
    ? 'No passages available.'
    : labelledPassages
      .map((passage) => {
        const headingDisplay = passage.headingPath.length > 1
          ? passage.headingPath.slice(1).join(' > ')
          : passage.headingPath[0] ?? passage.noteTitle;
        return `[${passage.label}] Note: ${passage.noteTitle}
Heading: ${headingDisplay}
Excerpt:
${passage.excerpt}
`;
      })
      .join('\n');

  const prompt = `${ANSWER_PROMPT_HEADER}

User question:
${userQuestion}

Normalised query:
${plan.normalized_query}

Passages:
${passagesSection}
`;

  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const repairSuffix = attempt === 0
      ? ''
      : `
The previous response was invalid JSON (${String(lastError).slice(0, 120)}). Respond with VALID JSON only.`;
    const raw = await model.chat(`${prompt}${repairSuffix}`, false, options.abortSignal);
    try {
      const jsonPayload = extractJsonObject(raw);
      const parsed = JSON.parse(jsonPayload);
      return validateAnswerPayload(parsed, labelledPassages.length);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Answer composer failed with unknown error');
}
