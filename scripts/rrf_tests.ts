import assert from 'assert';
import { reciprocalRankFusion, RrfQueryResults } from '../src/notes/rrf';

function testBasicFusion() {
  const queries: RrfQueryResults[] = [
    {
      label: 'normalized',
      items: [
        { id: 'note-1', score: 1.0 },
        { id: 'note-2', score: 0.8 },
        { id: 'note-3', score: 0.6 },
      ],
    },
    {
      label: 'expansions',
      items: [
        { id: 'note-3', score: 1.1 },
        { id: 'note-2', score: 1.0 },
      ],
    },
  ];

  const { results, logs } = reciprocalRankFusion(queries, { k: 60 });

  assert.strictEqual(results.length, 3);
  assert.strictEqual(results[0].id, 'note-3');
  assert.ok(results[0].fusedScore > results[1].fusedScore);
  assert.strictEqual(results[1].id, 'note-2');
  assert.strictEqual(results[2].id, 'note-1');

  assert.strictEqual(logs.length, 2);
  assert.strictEqual(logs[0].query, 'normalized');
  assert.strictEqual(logs[0].hits, 3);
  assert.deepStrictEqual(logs[0].topIds.slice(0, 2), ['note-1', 'note-2']);
}

function testMaxResultsAndDuplicates() {
  const queries: RrfQueryResults[] = [
    {
      label: 'normalized',
      items: [
        { id: 'note-1', score: 1.0 },
        { id: 'note-1', score: 0.9 },
        { id: 'note-2', score: 0.8 },
      ],
    },
    {
      label: 'entities',
      items: [
        { id: 'note-3', score: 1.2 },
        { id: 'note-2', score: 1.0 },
      ],
    },
  ];

  const { results } = reciprocalRankFusion(queries, { maxResults: 2, k: 10 });
  assert.strictEqual(results.length, 2);
  assert.ok(results.every((res) => ['note-2', 'note-3', 'note-1'].includes(res.id)));
  // Ensure duplicate within same query counted once
  const note1 = results.find((res) => res.id === 'note-1');
  if (note1) {
    assert.strictEqual(note1.contributions.length, 1);
  }
}

function main() {
  testBasicFusion();
  testMaxResultsAndDuplicates();
  // eslint-disable-next-line no-console
  console.log('RRF tests passed.');
}

main();
