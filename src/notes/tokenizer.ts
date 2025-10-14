const CJK_CHAR_CLASS = '\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u30FF\u31F0-\u31FF\uAC00-\uD7AF';
const THAI_CHAR_CLASS = '\u0E00-\u0E7F';

const CJK_SEQUENCE_RE = new RegExp(`[${CJK_CHAR_CLASS}]+`, 'gu');
const THAI_SEQUENCE_RE = new RegExp(`[${THAI_CHAR_CLASS}]+`, 'gu');
const CJK_CHAR_RE = new RegExp(`[${CJK_CHAR_CLASS}]`, 'u');
const THAI_CHAR_RE = new RegExp(`[${THAI_CHAR_CLASS}]`, 'u');
const ASCII_WORD_RE = /[a-z0-9]+(?:['-][a-z0-9]+)*/g;
const GENERIC_WORD_RE = /[\p{L}\p{N}]+(?:['-][\p{L}\p{N}]+)*/gu;

export type DetectedScript = 'cjk' | 'thai' | 'other';

export interface TokenizeResult {
  normalized: string;
  tokens: string[];
  script: DetectedScript;
}


/**
 * Normalize text for lexical matching: NFKD + strip diacritics + lowercase.
 * This routine intentionally retains ASCII apostrophes/hyphens inside words.
 */
export function normalizeText(input: string): string {
  if (!input) return '';
  const decomposed = input.normalize('NFKD');
  const stripped = decomposed.replace(/\p{M}/gu, '');
  const recomposed = stripped.normalize('NFKC');
  return recomposed.toLowerCase();
}


/**
 * Cheap check for CJK or Thai characters after normalization.
 * Used to decide whether to apply character bigram tokenization paths.
 */
export function containsCjkOrThai(input: string): boolean {
  const n = normalizeText(input);
  return CJK_CHAR_RE.test(n) || THAI_CHAR_RE.test(n);
}


function isAsciiAlphaNumeric(char: string): boolean {
  if (char.length === 0) return false;
  const code = char.charCodeAt(0);
  return (
    (code >= 48 && code <= 57) || // 0-9
    (code >= 65 && code <= 90) || // A-Z
    (code >= 97 && code <= 122)
  );
}

/**
 * Detect the dominant script family of a normalized string.
 * Guides tokenization strategy between CJK bigrams, Thai, and word segmentation.
 */
export function detectScript(normalized: string): DetectedScript {
  let cjkCount = 0;
  let thaiCount = 0;
  let latinCount = 0;

  for (const char of normalized) {
    if (CJK_CHAR_RE.test(char)) {
      cjkCount += 1;
    } else if (THAI_CHAR_RE.test(char)) {
      thaiCount += 1;
    } else if (isAsciiAlphaNumeric(char)) {
      latinCount += 1;
    }
  }

  if (cjkCount > 0 && cjkCount >= thaiCount) {
    return 'cjk';
  }
  if (thaiCount > 0) {
    return 'thai';
  }
  if (latinCount > 0) return 'other';
  return 'other';
}

function pushCharacterNGrams(sequence: string, out: string[]) {
  const chars = Array.from(sequence);
  if (chars.length === 0) return;
  if (chars.length === 1) {
    out.push(chars[0]);
    return;
  }
  if (chars.length === 2) {
    out.push(chars[0], chars[1], chars[0] + chars[1]);
    return;
  }
  for (let i = 0; i < chars.length - 1; i += 1) {
    out.push(chars[i] + chars[i + 1]);
  }
}

function collectAsciiWords(normalized: string): string[] {
  const matches = normalized.match(ASCII_WORD_RE);
  return matches ? matches.filter(Boolean) : [];
}

function tokenizeCjk(normalized: string): string[] {
  const tokens: string[] = [];
  let match: RegExpExecArray | null;
  CJK_SEQUENCE_RE.lastIndex = 0;
  while ((match = CJK_SEQUENCE_RE.exec(normalized)) !== null) {
    pushCharacterNGrams(match[0], tokens);
  }
  // Allow ASCII / digit sequences to surface alongside CJK bigrams.
  const asciiTokens = collectAsciiWords(normalized);
  tokens.push(...asciiTokens);
  return tokens;
}

function tokenizeThai(normalized: string): string[] {
  const tokens: string[] = [];
  let match: RegExpExecArray | null;
  THAI_SEQUENCE_RE.lastIndex = 0;
  while ((match = THAI_SEQUENCE_RE.exec(normalized)) !== null) {
    pushCharacterNGrams(match[0], tokens);
  }
  const asciiTokens = collectAsciiWords(normalized);
  tokens.push(...asciiTokens);
  return tokens;
}

function cleanWordToken(word: string): string | null {
  if (!word) return null;
  const matches = word.match(GENERIC_WORD_RE);
  if (!matches) return null;
  return matches[0] ?? null;
}

function tokenizeOther(normalized: string): string[] {
  const tokens: string[] = [];
  const segmenterAvailable =
    typeof Intl !== 'undefined' && typeof (Intl as Record<string, unknown>).Segmenter === 'function';

  if (segmenterAvailable) {
    const SegmenterCtor = (Intl as unknown as { Segmenter: new (...args: unknown[]) => any }).Segmenter;
    const segmenter = new SegmenterCtor('und', { granularity: 'word' });
    const segments: Iterable<{ segment: string; isWordLike?: boolean }> = segmenter.segment(normalized);
    for (const part of segments) {
      if (part.isWordLike === false) continue;
      const cleaned = cleanWordToken(part.segment);
      if (cleaned) tokens.push(cleaned);
    }
  } else {
    let match: RegExpExecArray | null;
    GENERIC_WORD_RE.lastIndex = 0;
    while ((match = GENERIC_WORD_RE.exec(normalized)) !== null) {
      if (match[0]) tokens.push(match[0]);
    }
  }

  return tokens;
}

/**
 * Tokenize free-form text into deterministic lexical tokens.
 * Returns the normalized string, the token list, and the detected script.
 */
export function tokenizeForSearch(input: string): TokenizeResult {
  const normalized = normalizeText(input);
  const script = detectScript(normalized);

  let tokens: string[] = [];
  if (script === 'cjk') {
    tokens = tokenizeCjk(normalized);
  } else if (script === 'thai') {
    tokens = tokenizeThai(normalized);
  } else {
    tokens = tokenizeOther(normalized);
  }

  return {
    normalized,
    tokens: tokens.filter((token) => token.length > 0),
    script,
  };
}
