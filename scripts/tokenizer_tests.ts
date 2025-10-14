import assert from 'assert';
import { detectScript, normalizeText, tokenizeForSearch } from '../src/notes/tokenizer';

function testNormalize() {
  const input = 'Café déjà-vu — Testing!';
  const expected = 'cafe deja-vu — testing!';
  assert.strictEqual(normalizeText(input), expected, 'Normalization should lowercase & strip diacritics');
}

function testLatinTokenization() {
  const { tokens, script } = tokenizeForSearch("Café déjà-vu — Testing 2025-08-01's summary.");
  assert.strictEqual(script, 'other');
  const tokenSet = new Set(tokens);
  assert.ok(tokenSet.has('cafe'), 'Expected token set to include "cafe"');
  assert.ok(tokenSet.has('testing'), 'Expected token set to include "testing"');
  assert.ok(tokenSet.has('summary'), 'Expected token set to include "summary"');
  const hasDejaCombined = tokenSet.has('deja-vu');
  const hasDejaSplit = tokenSet.has('deja') && tokenSet.has('vu');
  assert.ok(hasDejaCombined || hasDejaSplit, 'Expected deja tokens (combined or split)');
  const hasDateCombined = tokenSet.has("2025-08-01's");
  const hasDateSplit = tokenSet.has('2025') && tokenSet.has('08') && tokenSet.has('01');
  assert.ok(hasDateCombined || hasDateSplit, 'Expected date tokens (combined or split)');
}

function testCjkTokenization() {
  const single = tokenizeForSearch('我');
  assert.strictEqual(single.script, 'cjk');
  assert.deepStrictEqual(single.tokens, ['我']);

  const multi = tokenizeForSearch('我喜欢音乐');
  assert.strictEqual(multi.script, 'cjk');
  assert.deepStrictEqual(multi.tokens, ['我喜', '喜欢', '欢音', '音乐']);

  const mixed = tokenizeForSearch('AI研究2025');
  assert.strictEqual(mixed.script, 'cjk');
  assert.deepStrictEqual(mixed.tokens, ['研', '究', '研究', 'ai', '2025']);
}

function testThaiTokenization() {
  const thai = tokenizeForSearch('ประเทศไทย');
  assert.strictEqual(thai.script, 'thai');
  assert.ok(thai.tokens.length > 0);
  assert.ok(thai.tokens.every((token) => token.length >= 1));
}

function testCyrillicTokenization() {
  const cyrillic = tokenizeForSearch('Привет мир 123');
  assert.strictEqual(cyrillic.script, 'other');
  assert.deepStrictEqual(cyrillic.tokens, ['привет', 'мир', '123']);
}

function testDetectScript() {
  assert.strictEqual(detectScript('我喜欢音乐'), 'cjk');
  assert.strictEqual(detectScript('ประเทศไทย'), 'thai');
  assert.strictEqual(detectScript('hello world'), 'other');
}

function main() {
  testNormalize();
  testLatinTokenization();
  testCjkTokenization();
  testThaiTokenization();
  testCyrillicTokenization();
  testDetectScript();
  // eslint-disable-next-line no-console
  console.log('Tokenizer tests passed.');
}

main();
