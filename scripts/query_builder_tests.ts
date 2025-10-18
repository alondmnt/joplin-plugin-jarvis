import assert from 'assert';
import { CHAT_NOTES_PLAN_EXAMPLE } from '../src/prompts/chatWithNotes';
import { buildQueriesFromPlan } from '../src/notes/queryBuilder';

function testBuildQueries() {
  const plan = {
    ...CHAT_NOTES_PLAN_EXAMPLE,
    normalized_query: 'atlas mission milestones',
    expansions: ['launch rehearsal', 'post-launch checklist'],
    soft_terms: ['mission log'],
    acronyms: ['RCS', 'EVA'],
    entities: ['Project Atlas'],
    filters: {
      tags: ['projects/atlas', 'milestones'],
      notebooks: ['mission-control/2025'],
      created_after: '2025-10-01',
      created_before: '2025-10-31',
    },
  };

  const { queries, filters } = buildQueriesFromPlan(plan);

  // Filters should include tag/notebook/date clauses
  assert.ok(filters.includes('tag:projects/atlas'));
  assert.ok(filters.includes('tag:milestones'));
  assert.ok(filters.includes('notebook:mission-control/2025'));
  assert.ok(filters.includes('updated:20251001'));
  assert.ok(filters.includes('-updated:20251031'));

  const normalized = queries.find(q => q.label === 'normalized');
  assert.ok(normalized);
  assert.ok(normalized!.query.includes('atlas mission milestones'));
  assert.ok(normalized!.query.includes('tag:projects/atlas'));

  const expansions = queries.find(q => q.label === 'expansions');
  assert.ok(expansions);
  assert.ok(/any:1/.test(expansions!.query));
  assert.ok(expansions!.query.includes('"launch rehearsal"'));
  assert.ok(expansions!.query.includes('"post-launch checklist"'));
  assert.ok(expansions!.query.includes('"mission log"'));

  const titleQuery = queries.find(q => q.label === 'title');
  assert.ok(titleQuery);
  assert.ok(titleQuery!.query.includes('title:"atlas mission milestones"'));

  const acronyms = queries.find(q => q.label === 'acronyms');
  assert.ok(acronyms);
  assert.ok(/any:1/.test(acronyms!.query));
  assert.ok(acronyms!.query.includes('RCS'));

  const entities = queries.find(q => q.label === 'entities');
  assert.ok(entities);
  assert.ok(!/any:1/.test(entities!.query));
  assert.ok(entities!.query.includes('"Project Atlas"'));
}

function testEmptySections() {
  const plan = {
    ...CHAT_NOTES_PLAN_EXAMPLE,
    normalized_query: 'test query',
    expansions: [],
    soft_terms: [],
    acronyms: [],
    entities: [],
    filters: {
      tags: [],
      notebooks: [],
      created_after: null,
      created_before: null,
    },
  };
  const { queries } = buildQueriesFromPlan(plan);
  const labels = queries.map(q => q.label).sort();
  assert.deepStrictEqual(labels, ['normalized', 'title']);
}

function testSoftTermsFallback() {
  const plan = {
    ...CHAT_NOTES_PLAN_EXAMPLE,
    normalized_query: 'backup',
    expansions: [],
    soft_terms: ['redundant array', 'disk mirroring'],
    acronyms: [],
    entities: [],
    filters: { tags: [], notebooks: [], created_after: null, created_before: null },
  };
  const { queries } = buildQueriesFromPlan(plan);
  const expansions = queries.find(q => q.label === 'expansions');
  assert.ok(expansions, 'soft_terms should produce an expansions query');
  assert.ok(expansions!.query.startsWith('any:1'));
  assert.ok(expansions!.query.includes('"redundant array"'));
  assert.ok(expansions!.query.includes('"disk mirroring"'));
}

function main() {
  testBuildQueries();
  testEmptySections();
  testSoftTermsFallback();
  // eslint-disable-next-line no-console
  console.log('Query builder tests passed.');
}

main();
