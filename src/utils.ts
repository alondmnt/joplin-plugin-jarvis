export function with_timeout(msecs: number, promise: Promise<Response>): Promise<any> {
    const timeout = new Promise((resolve, reject) => {
      setTimeout(() => {
        reject(new Error("timeout"));
      }, msecs);
    });
    return Promise.race([timeout, promise]);
  }
