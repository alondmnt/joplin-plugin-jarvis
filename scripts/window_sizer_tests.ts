import assert from 'assert';
import { splitTextIntoTokenWindows } from '../src/notes/windowSizer';
import { encodingForModel } from 'js-tiktoken';

function testBasicSplit() {
  const options = { maxTokens: 4, strideRatio: 0.5, model: 'gpt-3.5-turbo' } as const;
  const windows = splitTextIntoTokenWindows('hello world this is a test of token windows', options);
  assert.ok(windows.length >= 2, 'expected multiple windows');
  windows.forEach((window) => {
    assert.ok(window.tokenCount <= options.maxTokens, 'window exceeds max tokens');
    assert.ok(window.start < window.end || window.tokenCount === 0);
  });
}

function testStrideAndCoverage() {
  const text = 'The Atlas mission notebook recorded launch rehearsals in October 2025. Milestone entries were logged the same week.';
  const options = { maxTokens: 10, strideRatio: 0.5, model: 'gpt-3.5-turbo' } as const;
  const encoder = encodingForModel(options.model);
  const tokens = encoder.encode(text);
  const strideTokens = Math.max(1, Math.floor(options.maxTokens * options.strideRatio));

  const windows = splitTextIntoTokenWindows(text, options);
  assert.ok(windows.length >= 2, 'expect overlap windows');
  assert.strictEqual(windows[0].start, 0);
  const lastWindow = windows[windows.length - 1];
  assert.strictEqual(lastWindow.end, tokens.length);

  for (let i = 1; i < windows.length; i += 1) {
    const prev = windows[i - 1];
    const current = windows[i];
    const expectedStart = Math.min(prev.start + strideTokens, tokens.length - current.tokenCount);
    assert.strictEqual(current.start, expectedStart);
  }
}

function testEmptyInput() {
  const windows = splitTextIntoTokenWindows('', { maxTokens: 5 });
  assert.deepStrictEqual(windows, []);
}

function main() {
  testBasicSplit();
  testStrideAndCoverage();
  testEmptyInput();
  // eslint-disable-next-line no-console
  console.log('Window sizer tests passed.');
}

main();
