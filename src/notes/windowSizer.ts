export interface TokenWindow {
  start: number;
  end: number;
  text: string;
  tokenCount: number;
}

export interface WindowSizerOptions {
  maxTokens: number;
  strideRatio?: number;
}

const DEFAULT_STRIDE_RATIO = 0.5;
const DEFAULT_TIKTOKEN_MODEL = 'gpt-3.5-turbo';
const FALLBACK_ENCODING = 'cl100k_base';

type Encoder = {
  encode: (text: string) => number[];
  decode: (tokens: number[]) => string;
};

let cachedEncoder: Encoder | null = null;

function getEncoder(): Encoder {
  if (cachedEncoder) {
    return cachedEncoder;
  }
  // Lazy import to avoid pulling the heavy encoder until needed.
  // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
  const { encodingForModel, getEncoding } = require('js-tiktoken');
  try {
    cachedEncoder = encodingForModel(DEFAULT_TIKTOKEN_MODEL);
    return cachedEncoder;
  } catch (error) {
    console.warn(
      `Failed to load tiktoken model "${DEFAULT_TIKTOKEN_MODEL}", defaulting to ${FALLBACK_ENCODING}`,
      error,
    );
    cachedEncoder = getEncoding(FALLBACK_ENCODING);
    return cachedEncoder;
  }
}

/**
 * Slice text into overlapping token windows sized for downstream LLM calls.
 * Uses js-tiktoken so counts stay consistent across desktop/mobile builds, with
 * a defensive fallback to `cl100k_base` when the requested model is unknown.
 */
export function splitTextIntoTokenWindows(text: string, options: WindowSizerOptions): TokenWindow[] {
  if (!text) return [];
  const encoder = getEncoder();
  const tokens = encoder.encode(text);
  if (tokens.length === 0) {
    return [
      {
        start: 0,
        end: 0,
        text,
        tokenCount: 0,
      },
    ];
  }

  const maxTokens = Math.max(1, options.maxTokens);
  const strideRatio = options.strideRatio ?? DEFAULT_STRIDE_RATIO;
  const strideTokens = Math.max(1, Math.floor(maxTokens * strideRatio));

  const windows: TokenWindow[] = [];
  let startTokenIndex = 0;

  while (startTokenIndex < tokens.length) {
    const endTokenIndex = Math.min(startTokenIndex + maxTokens, tokens.length);
    const sliceTokens = tokens.slice(startTokenIndex, endTokenIndex);
    const decoded = encoder.decode(sliceTokens);

    windows.push({
      start: startTokenIndex,
      end: endTokenIndex,
      text: decoded,
      tokenCount: sliceTokens.length,
    });

    if (endTokenIndex >= tokens.length) break;
    startTokenIndex += strideTokens;
  }

  return windows;
}
