import assert from 'assert';
import {
  CHAT_NOTES_PLAN_EXAMPLE,
  CHAT_NOTES_RERANK_EXAMPLE,
  validateChatNotesPlanResponse,
  validateChatNotesRerankResponse,
} from '../src/prompts/chatWithNotes';

function shouldThrow(fn: () => void, message: string) {
  let threw = false;
  try {
    fn();
  } catch (error) {
    threw = true;
    assert.ok(error instanceof Error, 'Expected thrown value to be an Error');
    return;
  }
  if (!threw) {
    throw new Error(message);
  }
}

function runPlanSchemaTests() {
  const parsed = validateChatNotesPlanResponse(CHAT_NOTES_PLAN_EXAMPLE);
  assert.strictEqual(parsed.prompt_version, 'chat-notes-plan-2');
  assert.ok(parsed.normalized_query.length > 0);

  shouldThrow(
    () => validateChatNotesPlanResponse({}),
    'Plan schema should reject empty objects',
  );

  shouldThrow(
    () =>
      validateChatNotesPlanResponse({
        ...CHAT_NOTES_PLAN_EXAMPLE,
        prompt_version: 'chat-notes-plan-1',
      }),
    'Plan schema should reject wrong version',
  );
}

function runRerankSchemaTests() {
  const parsed = validateChatNotesRerankResponse(CHAT_NOTES_RERANK_EXAMPLE);
  assert.strictEqual(parsed.prompt_version, 'chat-notes-rerank-2');
  assert.strictEqual(parsed.winner, 'A');

  shouldThrow(
    () => validateChatNotesRerankResponse({}),
    'Rerank schema should reject empty objects',
  );

  shouldThrow(
    () =>
      validateChatNotesRerankResponse({
        ...CHAT_NOTES_RERANK_EXAMPLE,
        winner: 'C',
      }),
    'Rerank schema should reject invalid winner values',
  );
}

function main() {
  runPlanSchemaTests();
  runRerankSchemaTests();
  // eslint-disable-next-line no-console
  console.log('Prompt contract tests passed.');
}

main();
