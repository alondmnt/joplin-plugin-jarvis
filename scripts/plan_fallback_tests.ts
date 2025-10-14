import assert from 'assert';
import { buildFallbackPlanFromText } from '../src/notes/planFallback';

function testFallbackPlan() {
  const text = 'The Atlas mission logged simulator rehearsals in October 2025 and documented launch milestones that week.';
  const plan = buildFallbackPlanFromText(text);
  assert.strictEqual(plan.prompt_version, 'chat-notes-plan-2');
  assert.ok(plan.normalized_query.length > 0);
  assert.ok(plan.hard_terms.length > 0);
  assert.ok(Array.isArray(plan.filters.tags));
}

function testFallbackAcronyms() {
  const text = 'Review the Hyper Text Transfer Protocol (HTTP) and Secure File Transfer Protocol (SFTP).';
  const plan = buildFallbackPlanFromText(text);
  assert.ok(plan.acronyms.includes('HTTP'));
  assert.ok(plan.acronyms.includes('SFTP'));
}

function main() {
  testFallbackPlan();
  testFallbackAcronyms();
  // eslint-disable-next-line no-console
  console.log('Fallback plan tests passed.');
}

main();
