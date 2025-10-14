import assert from 'assert';
import { LruCache } from '../src/notes/cache';

function testBasicSetGet() {
  const cache = new LruCache<string, number>({ maxSize: 2 });
  cache.set('a', 1);
  cache.set('b', 2);
  assert.strictEqual(cache.get('a'), 1);
  cache.set('c', 3);
  assert.strictEqual(cache.has('b'), false);
  assert.strictEqual(cache.get('c'), 3);
  assert.strictEqual(cache.get('a'), 1);
}

function testUpdateResetsOrder() {
  const cache = new LruCache<string, number>({ maxSize: 2 });
  cache.set('a', 1);
  cache.set('b', 2);
  cache.set('a', 10);
  cache.set('c', 3);
  assert.strictEqual(cache.has('a'), true);
  assert.strictEqual(cache.has('b'), false);
}

function testExpiry() {
  let current = 0;
  const clock = () => current;
  const cache = new LruCache<string, string>({ maxSize: 2, entryTtlMs: 1000, clock });
  cache.set('a', 'first');
  current = 500;
  assert.strictEqual(cache.get('a'), 'first');
  current = 1500;
  assert.strictEqual(cache.get('a'), undefined);
  assert.strictEqual(cache.has('a'), false);
}

function testClear() {
  const cache = new LruCache<string, number>({ maxSize: 2 });
  cache.set('x', 1);
  cache.set('y', 2);
  cache.clear();
  assert.strictEqual(cache.size, 0);
  assert.strictEqual(cache.get('x'), undefined);
}

function main() {
  testBasicSetGet();
  testUpdateResetsOrder();
  testExpiry();
  testClear();
  // eslint-disable-next-line no-console
  console.log('Cache tests passed.');
}

main();
