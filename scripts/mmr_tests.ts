import assert from 'assert';
import { MmrCandidate, selectWithMmr } from '../src/notes/passageScorer';

function makeCandidate(id: string, score: number, start: number, end: number): MmrCandidate<string> {
  return {
    id,
    score,
    window: { start, end },
    payload: id,
  };
}

function testSelectWithoutOverlap() {
  const candidates = [
    makeCandidate('a', 0.9, 0, 100),
    makeCandidate('b', 0.85, 200, 300),
    makeCandidate('c', 0.8, 400, 500),
  ];
  const selected = selectWithMmr(candidates, { lambda: 0.7, maxSelections: 2 });
  assert.deepStrictEqual(
    selected.map((c) => c.id),
    ['a', 'b'],
  );
}

function testOverlapSuppression() {
  const candidates = [
    makeCandidate('a', 0.9, 0, 200),
    makeCandidate('b', 0.85, 100, 250),
    makeCandidate('c', 0.8, 260, 360),
  ];
  const selected = selectWithMmr(candidates, { lambda: 0.7, maxSelections: 3, maxOverlapRatio: 0.5 });
  assert.deepStrictEqual(
    selected.map((c) => c.id),
    ['a', 'c'],
  );
}

function testLambdaInfluence() {
  const candidates = [
    makeCandidate('a', 0.9, 0, 200),
    makeCandidate('b', 0.89, 10, 210),
    makeCandidate('c', 0.6, 220, 300),
  ];
  const selectedHighLambda = selectWithMmr(candidates, { lambda: 0.9, maxSelections: 2, maxOverlapRatio: 0.4 });
  assert.deepStrictEqual(
    selectedHighLambda.map((c) => c.id),
    ['a', 'c'],
  );

  const selectedLowLambda = selectWithMmr(candidates, { lambda: 0.3, maxSelections: 2, maxOverlapRatio: 0.4 });
  assert.deepStrictEqual(
    selectedLowLambda.map((c) => c.id),
    ['a', 'c'],
  );
}

function main() {
  testSelectWithoutOverlap();
  testOverlapSuppression();
  testLambdaInfluence();
  // eslint-disable-next-line no-console
  console.log('MMR tests passed.');
}

main();
