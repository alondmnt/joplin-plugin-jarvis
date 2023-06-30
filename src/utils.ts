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
