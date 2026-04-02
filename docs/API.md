# Jarvis inter-plugin API

**version**: 1

Other Joplin plugins can call Jarvis commands to query the embedding index. All commands are invoked via `joplin.commands.execute()` and return plain JSON objects.

## conventions

- every response includes `ok: boolean`
- success: `{ ok: true, ... }`
- failure: `{ ok: false, error: string, message: string }`

`error` is a machine-readable code; `message` is a human-readable explanation.

## commands

### `jarvis.api.status`

Returns the current state of the Jarvis embedding index.

```js
const status = await joplin.commands.execute('jarvis.api.status');
```

#### response

| field | type | description |
|---|---|---|
| `ok` | `boolean` | always `true` on success |
| `version` | `number` | API version (currently 1) |
| `ready` | `boolean` | whether the embedding model is loaded and the index is queryable |
| `modelId` | `string \| null` | active embedding model identifier |
| `indexStats` | `object \| null` | `null` when not ready |
| `indexStats.noteCount` | `number` | notes in the index |
| `indexStats.blockCount` | `number` | text blocks in the index |

---

### `jarvis.api.search`

Semantic search over indexed notes. Provide a text query, a note ID (uses the note body as the query), or both (uses the text query in the context of the note).

```js
const result = await joplin.commands.execute('jarvis.api.search', {
  query: 'mitochondrial membrane potential',
  limit: 5,
  minSimilarity: 0.4,
});
```

#### parameters

Passed as a single object (first positional argument).

| param | type | required | description |
|---|---|---|---|
| `query` | `string` | one of `query` / `noteId` | free-text search query |
| `noteId` | `string` | one of `query` / `noteId` | Joplin note ID to use as query context |
| `limit` | `number` | no | max notes to return (overrides user setting `notes_max_hits`) |
| `minSimilarity` | `number` | no | 0-1 threshold (overrides user setting `notes_min_similarity`) |

#### response

```jsonc
{
  "ok": true,
  "results": [
    {
      "noteId": "abc123",
      "noteTitle": "mitochondria notes",
      "similarity": 0.82,
      "blocks": [
        {
          "title": "membrane potential",  // heading text
          "line": 14,                     // line number in note
          "level": 2,                     // heading level (0 = top)
          "similarity": 0.85,
          "text": "the actual block text extracted from the note body..."
        }
      ]
    }
  ]
}
```

**stable fields** (will not change within a major API version): `noteId`, `noteTitle`, `similarity`, `blocks[].title`, `blocks[].similarity`, `blocks[].text`.

**informational fields** (may change): `blocks[].line`, `blocks[].level`, `blocks[].bodyIdx`, `blocks[].length`. These reflect internal chunking offsets and are useful for debugging but should not be relied on for positioning.

Blocks are sorted by descending similarity within each note. Notes are sorted by descending aggregate similarity.

#### errors

| `error` code | meaning |
|---|---|
| `invalid_input` | neither `query` nor `noteId` was provided |
| `not_ready` | embedding model not loaded or index not built |
| `search_failed` | unexpected error during search (check Jarvis logs) |

## example: find notes related to the current note

```js
const note = await joplin.workspace.selectedNote();
const result = await joplin.commands.execute('jarvis.api.search', {
  noteId: note.id,
  limit: 10,
});
if (result.ok) {
  for (const r of result.results) {
    console.log(`${r.noteTitle} (${r.similarity.toFixed(2)})`);
  }
}
```
