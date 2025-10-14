import assert from 'assert';
import {
  DEFAULT_BM25L_PARAMS,
  anchorQuoteMatchCount,
  computeBm25LScore,
  contextNoteBoost,
  headingPathMatchCount,
  recencyBoost,
  spanProximityScore,
} from '../src/notes/passageScorer';

function almostEqual(actual: number, expected: number, epsilon = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `Expected ${expected}, received ${actual}`);
}

function testBm25LScore() {
  const params = { ...DEFAULT_BM25L_PARAMS, k1: 1.5, b: 0.75, delta: 0.5 };
  const score = computeBm25LScore({
    termFrequencies: { atlas: 3, milestones: 2 },
    documentLength: 100,
    averageDocumentLength: 120,
    totalDocuments: 100,
    documentFrequencies: { atlas: 10, milestones: 5 },
    queryTerms: ['atlas', 'milestones'],
    params,
  });

  const denom = (1 - params.b) + params.b * (100 / 120);
  const tfAtlas = (3 / denom) + params.delta;
  const tfMilestones = (2 / denom) + params.delta;
  const idfAtlas = Math.log(1 + (100 - 10 + 0.5) / (10 + 0.5));
  const idfMilestones = Math.log(1 + (100 - 5 + 0.5) / (5 + 0.5));
  const expected =
    idfAtlas * (((params.k1 + 1) * tfAtlas) / (params.k1 + tfAtlas)) +
    idfMilestones * (((params.k1 + 1) * tfMilestones) / (params.k1 + tfMilestones));

  almostEqual(score, expected);
}

function testBm25LZeroIdf() {
  const score = computeBm25LScore({
    termFrequencies: { common: 5 },
    documentLength: 80,
    averageDocumentLength: 100,
    totalDocuments: 20,
    documentFrequencies: { common: 20 },
    queryTerms: ['common'],
  });
  assert.strictEqual(score, 0);
}

function testSpanProximity() {
  const tokens = ['atlas', 'mission', 'milestones', 'logged', 'in', 'october'];
  const score = spanProximityScore(tokens, {
    hardTerms: ['atlas', 'october'],
    allTerms: ['atlas', 'october', 'mission'],
  });
  almostEqual(score, 2 / 6);

  const noMatch = spanProximityScore(tokens, {
    hardTerms: ['missing', 'terms'],
    allTerms: ['single'],
  });
  assert.strictEqual(noMatch, 0);
}

function testAnchorQuoteMatches() {
  const text = 'She wrote "multi-query fusion" in the conclusion and referenced "BM25L tweaks".';
  const count = anchorQuoteMatchCount(text, ['"multi-query fusion"', '"bm25l tweaks"', '"absent phrase"']);
  assert.strictEqual(count, 2);
}

function testHeadingPathMatches() {
  const count = headingPathMatchCount(['Projects', 'Atlas milestones', 'October 2025'], ['atlas', 'october', 'agents']);
  assert.strictEqual(count, 2);
}

function testContextNoteBoost() {
  const boost = contextNoteBoost('note-123', new Set(['note-001', 'note-123']));
  assert.strictEqual(boost, 1);
  assert.strictEqual(contextNoteBoost('note-999', new Set(['note-001'])), 0);
}

function testRecencyBoost() {
  const reference = new Date('2025-10-13T12:00:00Z');
  const recent = recencyBoost('2025-10-05T10:00:00Z', reference, 30);
  assert.ok(recent > 0.7);
  const stale = recencyBoost('2025-08-01T00:00:00Z', reference, 30);
  assert.strictEqual(stale, 0);
}

function main() {
  testBm25LScore();
  testBm25LZeroIdf();
  testSpanProximity();
  testAnchorQuoteMatches();
  testHeadingPathMatches();
  testContextNoteBoost();
  testRecencyBoost();
  // eslint-disable-next-line no-console
  console.log('Passage scorer tests passed.');
}

main();
