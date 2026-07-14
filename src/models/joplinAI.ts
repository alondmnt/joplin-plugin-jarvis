import joplin from 'api';
import type { ChatMessage, ChatMessageRole, ChatOptions } from 'api/types';
import { ModelError, truncateErrorForDialog } from '../utils';

// Minimum Joplin version that ships the joplin.ai plugin API (desktop beta).
const MIN_JOPLIN_MAJOR = 3;
const MIN_JOPLIN_MINOR = 7;

/** Compares a dotted version string against a major.minor floor. */
function isVersionAtLeast(version: string, major: number, minor: number): boolean {
  const parts = String(version ?? '').split('.');
  const vMajor = parseInt(parts[0], 10);
  const vMinor = parseInt(parts[1] ?? '0', 10);
  if (!Number.isFinite(vMajor) || !Number.isFinite(vMinor)) { return false; }
  if (vMajor !== major) { return vMajor > major; }
  return vMinor >= minor;
}

/**
 * Returns true when the running Joplin exposes the AI plugin API.
 *
 * We cannot feature-detect `joplin.ai` directly: the plugin sandbox wraps the
 * `joplin` global in a Proxy that fabricates a truthy value for *any* property
 * access, so `joplin.ai` and even `typeof joplin.ai.chat === 'function'` are
 * true regardless of support, failing only (across the IPC boundary) when
 * actually called. Instead we gate on the app version and platform via
 * `joplin.versionInfo()`, which returns real host data. The API is a
 * desktop-only beta introduced in Joplin 3.7. Any failure fails closed.
 */
export async function isAvailable(): Promise<boolean> {
  try {
    const info = await joplin.versionInfo();
    if (!info || info.platform !== 'desktop') { return false; }
    return isVersionAtLeast(info.version, MIN_JOPLIN_MAJOR, MIN_JOPLIN_MINOR);
  } catch {
    return false;
  }
}

/**
 * Sends a chat conversation to the model configured in Joplin's own settings
 * (Settings → AI) and returns the assistant's text. Jarvis does not own the
 * provider or API key here — Joplin does — so the failure modes are mostly
 * configuration issues (AI disabled, remote access not allowed, missing key)
 * that the user resolves in Joplin, not in Jarvis. On error we surface a
 * dialog pointing there, then retry on OK or throw a ModelError on cancel
 * (mirroring the OpenAI/Hugging Face paths).
 *
 * @param prompt chat entries with roles already normalised to
 *   system/user/assistant by the caller
 * @param temperature sampling temperature on Joplin's 0-1 scale, or
 *   null/undefined to take the provider default
 */
export async function query_chat(
    prompt: Array<{role: string; content: string;}>,
    temperature: number): Promise<string> {

  const messages: ChatMessage[] = prompt.map((message) => ({
    role: message.role as ChatMessageRole,
    content: message.content,
  }));

  const options: ChatOptions = {};
  if (temperature !== null && temperature !== undefined) {
    options.temperature = temperature;
  }

  let error_message: string | null = null;
  try {
    const result = await joplin.ai.chat(messages, options);
    if (result && typeof result.text === 'string') {
      return result.text;
    }
    error_message = 'Joplin AI returned an empty response';

  } catch (error) {
    error_message = error instanceof Error ? error.message : String(error);
  }

  // display error message (truncated for dialog, full message logged)
  console.error(`Joplin AI chat error: ${error_message}`);
  const errorHandler = await joplin.views.dialogs.showMessageBox(
    `Error from Joplin AI: ${truncateErrorForDialog(error_message)}\n\n` +
    `Check that AI is enabled and a provider is configured in ` +
    `Joplin → Settings → AI (and that remote access is allowed for cloud ` +
    `providers). Press OK to retry.`
  );

  // cancel button
  if (errorHandler === 1) {
    throw new ModelError(`Joplin AI chat failed: ${error_message}`);
  }

  // retry
  return await query_chat(prompt, temperature);
}
