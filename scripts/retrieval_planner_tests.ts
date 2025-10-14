import assert from 'assert';
import { ChatNotesPlanTurn } from '../src/prompts/chatWithNotes';
import { generatePlan, clearPlannerCache } from '../src/notes/retrievalPlanner';

class StubModel {
  public prompts: string[] = [];
  private responses: string[];

  constructor(responses: string[]) {
    this.responses = responses;
  }

  async chat(prompt: string): Promise<string> {
    this.prompts.push(prompt);
    if (this.responses.length === 0) {
      throw new Error('No stub response available');
    }
    return this.responses.shift();
  }
}

const baseInput = {
  scopeSummary: 'Notebook: Project Atlas',
  citedNotes: [{ id: 'atlas-1', title: 'Atlas Mission Log' }],
  conversation: [
    { role: 'user', content: 'What milestones did we record in October?' } as ChatNotesPlanTurn,
  ],
};

const validPlanJson = `assistant:
{
  "prompt_version": "chat-notes-plan-2",
  "normalized_query": "atlas mission october milestones",
  "expansions": ["atlas", "mission", "milestones"],
  "hard_terms": ["atlas"],
  "soft_terms": ["milestones"],
  "entities": ["Project Atlas"],
  "acronyms": [],
  "filters": {
    "tags": [],
    "notebooks": [],
    "created_after": null,
    "created_before": null
  },
  "context_notes": []
}
user:`;

async function testValidPlanCaching() {
  clearPlannerCache();
  const model = new StubModel([validPlanJson]);
  const first = await generatePlan(model, baseInput);
  assert.strictEqual(first.usedFallback, false);
  assert.strictEqual(first.repaired, false);
  assert.strictEqual(first.fromCache, false);
  assert.strictEqual(first.plan.normalized_query, 'atlas mission october milestones');
  const second = await generatePlan(model, baseInput);
  assert.strictEqual(second.fromCache, true);
  assert.strictEqual(model.prompts.length, 1);
}

async function testRepairAttempt() {
  clearPlannerCache();
  const invalid = 'assistant: Not JSON user:';
  const repairedModel = new StubModel([invalid, validPlanJson]);
  const result = await generatePlan(repairedModel, baseInput);
  assert.strictEqual(result.repaired, true);
  assert.strictEqual(result.usedFallback, false);
  assert.ok(result.rawResponse.includes('Not JSON') === false);
}

async function testFallback() {
  clearPlannerCache();
  const model = new StubModel(['assistant: nope user:', 'assistant: still nope user:']);
  const result = await generatePlan(model, baseInput, { disableCache: true });
  assert.strictEqual(result.usedFallback, true);
  assert.ok(result.plan.normalized_query.length > 0);
}

async function main() {
  await testValidPlanCaching();
  await testRepairAttempt();
  await testFallback();
  // eslint-disable-next-line no-console
  console.log('Retrieval planner tests passed.');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
