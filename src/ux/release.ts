export const RELEASE_NOTES = {
  version: 'v0.14.0',
  notes: `v0.14.0:
- improved: upgraded models (gpt-5.6, claude-opus-5, claude-sonnet-5, gemini-3.7-flash)
  - the default chat model is now gpt-5.6-luna
- improved: one tunable context budget replaces the built-in per-model table
  - Chat: Max tokens renamed to Chat: Max context tokens
- improved: sampling settings left to the model where it rejects them (Claude)
- improved: clearer error messages from Google endpoints
- fixed: Gemini through a custom OpenAI-compatible endpoint failed on every chat
- fixed: annotation could overwrite a note title using an empty note
- fixed: Note mode now says when a long note was truncated
- fixed: custom models named olmo, orca or openhermes lost the system message

Full changelog: https://github.com/alondmnt/joplin-plugin-jarvis/releases
`,
};
