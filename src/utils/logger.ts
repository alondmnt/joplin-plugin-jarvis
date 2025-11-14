export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface Logger {
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

const prefix = '[Jarvis]';

const defaultLogger: Logger = {
  debug(message: string, ...args: unknown[]) {
    console.debug(prefix, message, ...args);
  },
  info(message: string, ...args: unknown[]) {
    console.info(prefix, message, ...args);
  },
  warn(message: string, ...args: unknown[]) {
    console.warn(prefix, message, ...args);
  },
  error(message: string, ...args: unknown[]) {
    console.error(prefix, message, ...args);
  },
};

export function getLogger(): Logger {
  return defaultLogger;
}

