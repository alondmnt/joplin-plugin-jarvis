import assert from 'assert';
import { phraseMatch } from '../src/notes/phraseMatcher';

function testLatinPhraseMatch() {
  assert.strictEqual(phraseMatch('Casey loves music festivals', 'music festivals'), true);
  assert.strictEqual(phraseMatch('Casey loves music festivals', 'science fair'), false);
}

function testCjkPhraseMatch() {
  const text = '我喜欢音乐节并且记录了所有演出';
  assert.strictEqual(phraseMatch(text, '音乐节'), true);
  assert.strictEqual(phraseMatch(text, '演出'), true);
  assert.strictEqual(phraseMatch(text, '博物馆'), false);
}

function testThaiPhraseMatch() {
  const text = 'ฉันรักดนตรีและงานเทศกาล';
  assert.strictEqual(phraseMatch(text, 'ดนตรี'), true);
  assert.strictEqual(phraseMatch(text, 'เทศกาล'), true);
  assert.strictEqual(phraseMatch(text, 'สัปดาห์หนังสือ'), false);
}

function main() {
  testLatinPhraseMatch();
  testCjkPhraseMatch();
  testThaiPhraseMatch();
  // eslint-disable-next-line no-console
  console.log('Phrase matcher tests passed.');
}

main();
