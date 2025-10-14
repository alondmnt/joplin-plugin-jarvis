import assert from 'assert';
import { buildAcronym, extractAcronyms, extractCapitalizedEntities, extractTopKeywords } from '../src/notes/textUtils';

function testBuildAcronym() {
  assert.strictEqual(buildAcronym('Hyper Text Transfer Protocol'), 'HTTP');
  assert.strictEqual(buildAcronym('central processing unit'), 'CPU');
  assert.strictEqual(buildAcronym('hi'), '');
}

function testExtractAcronyms() {
  const text = 'Review the HTTP spec and the GPU driver update.';
  const acronyms = extractAcronyms(text);
  assert.ok(acronyms.includes('HTTP'));
  assert.ok(acronyms.includes('GPU'));
}

function testExtractEntities() {
  const entities = extractCapitalizedEntities('Morgan met Riley in New York with Quinn.', 3);
  assert.deepStrictEqual(entities, ['Morgan', 'Riley', 'New']);
}

function testTopKeywords() {
  const { hard, soft } = extractTopKeywords(
    'atlas atlas mission mission rehearsal checklist checklist',
    2,
    3,
  );
  assert.ok(hard.length <= 2);
  assert.ok(hard[0].length > 0);
  assert.ok(soft.includes('rehearsal'));
}

function main() {
  testBuildAcronym();
  testExtractAcronyms();
  testExtractEntities();
  testTopKeywords();
  // eslint-disable-next-line no-console
  console.log('Acronym/text utils tests passed.');
}

main();
