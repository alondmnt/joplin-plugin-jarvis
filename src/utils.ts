import joplin from 'api';
import { Readability } from "@mozilla/readability";
const JSDOMParser = require("@mozilla/readability/JSDOMParser");
const createDOMPurify = require("dompurify");
const TurndownModule = require("turndown");
const turndownGfmModule = require("turndown-plugin-gfm");

export function with_timeout(msecs: number, promise: Promise<any>): Promise<any> {
  const timeout = new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error("timeout"));
    }, msecs);
  });
  return Promise.race([timeout, promise]);
}

interface TimeoutOptions {
  interactive?: boolean;
}

export async function timeout_with_retry(
    msecs: number,
    promise_func: () => Promise<any>,
    default_value: any = '',
    options: TimeoutOptions = {}): Promise<any> {

  const interactive = options.interactive ?? true;
  try {
    return await with_timeout(msecs, promise_func());
  } catch (error) {
    console.log(error);
    if (error.message.toLowerCase().includes('timeout')) {
      if (!interactive) {
        throw new ModelError(`Request timeout (${msecs / 1000} sec).`);
      }
      const choice = await joplin.views.dialogs.showMessageBox(`Error: Request timeout (${msecs / 1000} sec).\nPress OK to retry.`);
      if (choice === 0) {
        // OK button
        return await timeout_with_retry(msecs, promise_func);
      }
      // Cancel button
      throw new ModelError('Operation cancelled by user');
    }
    // For all other errors, propagate them unchanged
    throw error;
  }
}

// provide a text split into paragraphs, sentences, words, etc,
// or even a complete [text] (and then split_by is used).
// return a 2D array where in each row the total token sum is
// less than max_tokens.
// optionally, select from the end of the text (prefer = 'last').
export function split_by_tokens(
  parts: Array<string>,
  model: { count_tokens: (text: string) => number },
  max_tokens: number,
  prefer: string = 'first',
  split_by: string = ' ',  // can be null to split by characters
): Array<Array<string>> {

  // preprocess parts to ensure each part is smaller than max_tokens
  function preprocess(part: string): Array<string> {
    const token_count = model.count_tokens(part);

    if (token_count <= max_tokens) { return [part]; }

    // split the part in half
    let part_arr: any = part;
    const use_regex = (split_by !== null) &&
                      (part.split(split_by).length > 1);
    if (use_regex) {
      part_arr = part_arr.split(split_by);
    }

    const middle = Math.floor(part_arr.length / 2);
    let left_part = part_arr.slice(0, middle);
    let right_part = part_arr.slice(middle);

    if (use_regex) {
      left_part = left_part.join(split_by);
      right_part = right_part.join(split_by);
    }

    const left_split = preprocess(left_part);
    const right_split = preprocess(right_part);

    return [...left_split, ...right_split];
  }

  const small_parts = parts.map(preprocess).flat();

  // get the token sum of each text
  const token_counts = small_parts.map(text => model.count_tokens(text));
  if (prefer === 'last') {
    token_counts.reverse();
    small_parts.reverse();
  }

  // merge parts until the token sum is greater than max_tokens
  let selected: Array<Array<string>> = [];
  let token_sum = 0;
  let current_selection: Array<string> = [];

  for (let i = 0; i < token_counts.length; i++) {
    if (token_sum + token_counts[i] > max_tokens) {
      // return the accumulated texts based on the prefer option
      if (prefer === 'last') {
        current_selection.reverse();
      }
      selected.push(current_selection);
      current_selection = [];
      token_sum = 0;
    }

    current_selection.push(small_parts[i]);
    token_sum += token_counts[i];
  }

  if (current_selection.length > 0) {
    // return the accumulated texts based on the prefer option
    if (prefer === 'last') {
      current_selection.reverse();
    }
    selected.push(current_selection);
  }

  return selected;
}

export async function consume_rate_limit(
    model: { requests_per_second: number, request_queue: Array<any>, last_request_time: number }) {
  /*
    1. Each embed() call creates a request_promise and adds a request object to the requestQueue.
    2. The consume_rate_limit() method is called for each embed() call.
    3. The consume_rate_limit() method checks if there are any pending requests in the requestQueue.
    4. If there are pending requests, the method calculates the necessary wait time based on the rate limit and the time elapsed since the last request.
    5. If the calculated wait time is greater than zero, the method waits using setTimeout() for the specified duration.
    6. After the wait period, the method processes the next request in the requestQueue by shifting it from the queue and resolving its associated promise.
    7. The resolved promise allows the corresponding embed() call to proceed further and generate the embedding for the text.
    8. If there are additional pending requests in the requestQueue, the consume_rate_limit() method is called again to handle the next request in the same manner.
    9. This process continues until all requests in the requestQueue have been processed.
  */
  const now = Date.now();
  const time_elapsed = now - model.last_request_time;

  // calculate the time required to wait between requests
  const wait_time = model.request_queue.length * (1000 / model.requests_per_second);

  if (time_elapsed < wait_time) {
    await new Promise((resolve) => setTimeout(resolve, wait_time - time_elapsed));
  }

  model.last_request_time = now;

  // process the next request in the queue
  if (model.request_queue.length > 0) {
    const request = model.request_queue.shift();
    request.resolve(); // resolve the request promise
  }
}

export function search_keywords(text: string, query: string): boolean {
  // split the query into words/phrases
  const parts = preprocess_query(query).match(/"[^"]+"|\S+/g) || [];

  // build regular expression patterns for words/phrases
  const patterns = parts.map(part => {
      if (part.startsWith('"') && part.endsWith('"')) {
        // match exact phrase
        return `(?=.*\\b${part.slice(1, -1)}\\b)`;
      } else if (part.endsWith('*')) {
        // match prefix (remove the '*' and don't require a word boundary at the end)
        return `(?=.*\\b${part.slice(0, -1)})`;
      } else {
        // match individual keywords
        return `(?=.*\\b${part}\\b)`;
      }
  });

  // combine patterns into a single regular expression
  const regex = new RegExp(patterns.join(''), 'is');

  // return true if all keywords/phrases are found, false otherwise
  return regex.test(text);
}

function preprocess_query(query: string) {
  const operators = [
    'any', 'title', 'body', 'tag', 'notebook',
    'created', 'updated', 'due', 'type', 'iscompleted',
    'latitude', 'longitude', 'altitude', 'resource',
    'sourceurl', 'id'
  ];

  // build a regex pattern to match <operator>:<keyword>
  const regexPattern = new RegExp(`\\b(?:${operators.join('|')}):\\S+`, 'g');

  // remove <operator>:<keyword> patterns from the query
  return query.replace(regexPattern, '').trim();
}

export async function get_all_tags(): Promise<Array<string>> {
  // TODO: get only *used* tags based on all notes
  const tags: Array<string> = [];
  let page = 0;
  let some_tags: any;

  do {
    page += 1;
    some_tags = await joplin.data.get(['tags'], { fields: ['title'], page: page });

    tags.push(...some_tags.items.map((tag: any) => tag.title));
    
    const hasMore = some_tags.has_more;
    // Clear API response to help GC
    clearApiResponse(some_tags);
    
    if (!hasMore) break;
  } while(true);

  return tags;
}

export function escape_regex(string: string): string {
  return string
    .replace(/---/g, '')  // ignore dividing line
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    .trim();
}

// replace the last occurrence of a pattern in a string
export function replace_last(str: string, pattern: string, replacement: string): string {
  const index = str.lastIndexOf(pattern);
  if (index === -1) return str;  // Pattern not found, return original string

  // Construct the new string
  return str.substring(0, index) + replacement + str.substring(index + pattern.length);
}

// Custom error for user cancellation
export class ModelError extends Error {
  constructor(message: string = 'Model error') {
    super(message);
    this.name = 'ModelError';
  }
}

/**
 * Truncate long error messages for display in dialogs.
 * Shows the first N chars, "....", and the last N chars.
 * Full message should be logged separately.
 */
export function truncateErrorForDialog(message: string, maxChars: number = 200): string {
  if (!message || message.length <= maxChars) {
    return message;
  }
  const halfLen = Math.floor((maxChars - 4) / 2);  // 4 chars for "...."
  const start = message.slice(0, halfLen);
  const end = message.slice(-halfLen);
  return `${start}....${end}`;
}

/**
 * Convert messy HTML -> normalized "Markdown-lite" that you use for BOTH embeddings and LLM context.
 * Deterministic, compact, and keeps helpful structure (headings, lists, code, TSV tables).
 */
export async function htmlToText(html: string): Promise<string> {
  const doc = createDocument(html);
  const article = doc ? new Readability(doc).parse() : null;
  const contentHtml = article?.content || html;

  const clean = sanitizeHtml(contentHtml, doc);

  const TurndownCtor = getTurndownConstructor();
  if (!TurndownCtor) {
    console.warn("htmlToText: TurndownService unavailable; returning plain text fallback");
    return stripHtml(clean);
  }

  const td = new TurndownCtor({ codeBlockStyle: "fenced", bulletListMarker: "-" });
  const gfm = getGfmPlugin();
  if (gfm) {
    td.use(gfm);
  } else {
    console.warn("htmlToText: turndown GFM plugin unavailable; continuing without it");
  }

  let md = td.turndown(clean);

  // Normalize to “Markdown-lite”
  md = md
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)")        // links
    .replace(
      /((?:^\s*\|.*\|\s*$\n?)+)/gm,                        // GFM tables -> TSV fenced
      blk => {
        const lines = blk.trim().split("\n")
          .map(l => l.replace(/^\s*\||\|\s*$/g, ""))
          .map(l => l.split("|").map(c => c.trim()).join("\t"))
          .join("\n");
        return "```\n" + lines + "\n```";
      }
    )
    .replace(/^\s*>\s?/gm, "")                             // blockquotes
    .replace(/[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return md;
}

function createDocument(html: string): any | null {
  if (typeof DOMParser !== 'undefined') {
    try {
      return new DOMParser().parseFromString(html, "text/html");
    } catch (error) {
      console.warn("htmlToText: DOMParser failed, falling back to JSDOMParser", error);
    }
  }

  try {
    const parser = new JSDOMParser();
    return parser.parse(html);
  } catch (error) {
    console.warn("htmlToText: unable to parse HTML", error);
    return null;
  }
}

function sanitizeHtml(html: string, doc: any): string {
  const purifier = getDOMPurifyInstance(doc);
  const options = {
    ALLOWED_TAGS: [
      "h1","h2","h3","h4","h5","h6","p","ul","ol","li",
      "pre","code","table","thead","tbody","tr","th","td","a","strong","em","img","br"
    ],
    ALLOWED_ATTR: ["href","alt"],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
    FORBID_ATTR: ["style","onerror","onclick","id","class"]
  };

  if (purifier?.sanitize) {
    return purifier.sanitize(html, options);
  }

  console.warn("htmlToText: DOMPurify sanitization unavailable; falling back to basic cleaning");
  return basicSanitize(html);
}

function getDOMPurifyInstance(doc: any): any {
  const module: any = (createDOMPurify as any).default ?? createDOMPurify;

  if (module?.sanitize) {
    return module;
  }

  if (typeof module === 'function') {
    const windowLike = (typeof window !== 'undefined' ? window : undefined) || doc?.defaultView;
    if (windowLike) {
      try {
        return module(windowLike);
      } catch (error) {
        console.warn("htmlToText: DOMPurify factory failed", error);
      }
    }
  }

  return null;
}

function basicSanitize(html: string): string {
  return html
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?>[\s\S]*?<\/style>/gi, '');
}

function getTurndownConstructor(): any {
  const mod: any = TurndownModule;
  if (typeof mod === 'function') { return mod; }
  if (typeof mod?.default === 'function') { return mod.default; }
  if (typeof mod?.TurndownService === 'function') { return mod.TurndownService; }
  return null;
}

function getGfmPlugin(): ((service: any) => void) | null {
  const mod: any = turndownGfmModule;
  if (!mod) { return null; }
  if (typeof mod === 'function') { return mod; }
  if (typeof mod?.gfm === 'function') { return mod.gfm; }
  if (typeof mod?.default === 'function') { return mod.default; }
  return null;
}

function stripHtml(html: string): string {
  return html
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Aggressively clear object references to help GC release memory.
 * Particularly useful after processing large note bodies or API results.
 * 
 * @param obj - Object, array, or primitive to clear
 * @param visited - Internal tracking to prevent infinite recursion
 * @returns null to allow convenient reassignment (e.g., `obj = clearRefs(obj)`)
 */
export function clearObjectReferences<T extends Record<string, any>>(
  obj: T | null | undefined,
  visited: WeakSet<object> = new WeakSet()
): null {
  // Skip nullish, non-objects, or already visited
  if (!obj || typeof obj !== 'object') {
    return null;
  }
  if (visited.has(obj)) {
    return null;
  }
  visited.add(obj);

  try {
    if (Array.isArray(obj)) {
      // Clear array elements (faster than delete for large arrays)
      obj.length = 0;
    } else if (obj instanceof Map) {
      obj.clear();
    } else if (obj instanceof Set) {
      obj.clear();
    } else {
      // Clear own properties (not inherited)
      const keys = Object.keys(obj);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        try {
          delete obj[key];
        } catch {
          // Ignore readonly/non-configurable properties
        }
      }
    }
  } catch (error) {
    // Silently ignore errors - GC will eventually handle it
  }
  return null;
}

/**
 * Clear API response objects that may hold large note bodies.
 * Clears the items array and common pagination fields.
 */
export function clearApiResponse(response: any): null {
  if (!response || typeof response !== 'object') {
    return null;
  }

  try {
    // Clear items array if present
    if (Array.isArray(response.items)) {
      response.items.length = 0;
    }
    // Clear common fields
    delete response.items;
    delete response.has_more;
  } catch {
    // Ignore errors
  }

  return null;
}

// Module-level regex patterns (created once, reused on every call)
// Following existing pattern in chat.ts:61
const jarvisSummaryPattern = /<!-- jarvis-summary-start -->[\s\S]*?<!-- jarvis-summary-end -->/g;
const jarvisLinksPattern = /<!-- jarvis-links-start -->[\s\S]*?<!-- jarvis-links-end -->/g;
const jarvisCmdPattern = /```jarvis[\s\S]*?```/gm;

/**
 * Strip all Jarvis-generated blocks from text.
 * Removes: summary blocks, links blocks, and jarvis command blocks.
 */
export function stripJarvisBlocks(text: string): string {
  return text
    .replace(jarvisSummaryPattern, '')
    .replace(jarvisLinksPattern, '')
    .replace(jarvisCmdPattern, '');
}
