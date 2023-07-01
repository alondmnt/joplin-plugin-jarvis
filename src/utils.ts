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
export async function split_by_tokens(
  parts: Array<string>,
  model: { count_tokens: (text: string) => Promise<number> },
  max_tokens: number,
  prefer: string = 'first',
  split_by: string = ' ',  // can be null to split by characters
): Promise<Array<Array<string>>> {

  // preprocess parts to ensure each part is smaller than max_tokens
  async function preprocess(part: string): Promise<Array<string>> {
    const token_count = await model.count_tokens(part);

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

    const left_split = await preprocess(left_part);
    const right_split = await preprocess(right_part);

    return [...left_split, ...right_split];
  }

  const small_parts = (await Promise.all(parts.map(preprocess))).flat();

  // get the token sum of each text
  const token_counts = await Promise.all(small_parts.map(async (text) => model.count_tokens(text)));
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
