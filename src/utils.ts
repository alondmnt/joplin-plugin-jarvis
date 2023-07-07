import joplin from 'api';

export function with_timeout(msecs: number, promise: Promise<any>): Promise<any> {
  const timeout = new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error("timeout"));
    }, msecs);
  });
  return Promise.race([timeout, promise]);
}

export async function timeout_with_retry(msecs: number,
    promise_func: () => Promise<any>, default_value: any = ''): Promise<any> {
  try {
    return await with_timeout(msecs, promise_func());
  } catch (error) {
    const choice = await joplin.views.dialogs.showMessageBox(`Error: Request timeout (${msecs / 1000} sec).\nPress OK to retry.`);
    if (choice === 0) {
      // OK button
      return await timeout_with_retry(msecs, promise_func);
    }
    // Cancel button
    return default_value;
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
