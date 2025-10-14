import { normalizeText } from './tokenizer';

function containsCjkOrThai(input: string): boolean {
  return /[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\u31F0-\u31FF\uAC00-\uD7AF\u0E00-\u0E7F]/u.test(input);
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Optional: only keep if you really want a last-ditch heuristic
function toBigrams(text: string): string[] {
  const chars = Array.from(text);
  if (chars.length <= 1) return chars;
  const grams: string[] = [];
  for (let i = 0; i < chars.length - 1; i++) grams.push(chars[i] + chars[i + 1]);
  return grams;
}

/**
 * Check whether a phrase appears in text, with special handling for CJK/Thai.
 * Falls back to a gap-tolerant regex so we can spot phrases without spaces.
 */
export function phraseMatch(text: string, phrase: string, maxGap = 6): boolean {
  if (!phrase || !text) return false;

  const normalizedText = normalizeText(text);
  const normalizedPhrase = normalizeText(phrase);
  if (!normalizedPhrase) return false;

  // Fast path: exact substring for all scripts
  if (normalizedText.includes(normalizedPhrase)) return true;

  // If phrase has no CJK/Thai, be strict: no fuzzy match for non-CJK phrases
  if (!containsCjkOrThai(normalizedPhrase)) {
    return false;
  }

  // CJK/Thai: allow up to `maxGap` arbitrary chars between successive phrase chars
  // Build a regex like: 你.{0,6}好.{0,6}世.{0,6}界
  const chars = Array.from(normalizedPhrase);
  if (chars.length === 1) {
    return normalizedText.includes(chars[0]);
  }
  const pattern = chars.map(escapeRegex).join(`.{0,${Math.max(0, maxGap)}}`);
  const re = new RegExp(pattern, 'u');
  if (re.test(normalizedText)) return true;

  // Optional last fallback (heuristic, may false-positive): bigram coverage
  // const phraseBigrams = new Set(toBigrams(normalizedPhrase));
  // const textBigrams = new Set(toBigrams(normalizedText));
  // return [...phraseBigrams].every(bg => textBigrams.has(bg));

  return false;
}
