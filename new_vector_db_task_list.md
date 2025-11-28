# Jarvis UserData Embeddings — Engineering Task List

Task list derived from `new_vector_db_prd.md`. Grouped roughly in execution order; adjust sequencing as dependencies evolve. Add docstrings to functions and comments for non-trivial lines. Use snake_case for function names.

## 0. Coordination & Feature Flag
- [x] Enable gated rollout via `notes_db_in_user_data` (phase flag, config plumbing, defaults off).
- [x] Capture open-question answers (userData size ceiling, E2EE guarantees, API limits, centroid policy) in `notes/new_vector_db_open_questions.md` and keep updated.
- [x] Confirm upgrade plan aligns with current database implementation constraints; document any deviations. *(See `migration_coordination.md` - **APPROVED**: No conflicts, ready for implementation)*

## 1. Data Model & Persistence Layer
- [x] Define shared TypeScript types for `NoteEmbMeta`, `BlockRowMeta`, `EmbShard`, and `EmbStore` interfaces.
- [x] Implement `EmbStore` backed by Joplin `userDataGet/Set/Delete`, including stable key construction (`jarvis/v1/meta`, `jarvis/v1/emb/<modelId>/live/<i>`).
- [x] Add base64 helpers to encode/decode q8 vectors, Float32 scales, and Uint16 centroid ids.
- [x] Support epoch-based two-phase commit (write shards → commit `meta.current` → prune legacy shard slots).
- [x] Persist bounded `history` entries and implement `gcOld` to delete stale shards or rewrite them as empty.
- [x] Cache per-note `meta` in memory (LRU) with invalidation on writes to reduce read round-trips.

## 1.5. Metadata Structure Refinements & Multi-Model Support
- [x] **Update tag naming conventions:**
  - **Catalog tag**: Change from `jarvis.database` to `jarvis-database` (breaking change, do not support old tag).
    - Update write path to add `jarvis-database`.
  - **Exclusion tag**: Change from `exclude.from.jarvis` to `jarvis-exclude` (but maintain backward compatibility).
    - Update `EXCLUDE_TAG` constant in `catalog.ts` and `embeddings.ts` (lines 420-422).
    - Check logic: `note_tags.includes('jarvis-exclude') || note_tags.includes('exclude.from.jarvis')`
    - Support both tags for backward compatibility (users may have notes already tagged with old format).
    - Document in user-facing docs: "Use `jarvis-exclude` (old `exclude.from.jarvis` still supported)".
- [x] **Refactor `NoteEmbMeta` to support multiple models per note:**
  - Replace flat structure with `models: { [modelId: string]: ModelMetadata }` to store metadata per model.
  - Each model entry contains: `dim`, `modelVersion`, `embeddingVersion`, `settings` (explicit), and `current` state.
  - Devices read embeddings using their local `settings.notes_model` as the key (not synced across devices).
  - Remove the top-level `modelId`, `dim`, `metric`, `modelVersion`, `embeddingVersion`, `maxBlockSize`, `settingsHash` fields (now per-model).
- [x] **Replace `settingsHash` with explicit `EmbeddingSettings` object:**
  - Define `EmbeddingSettings` interface with: `embedTitle`, `embedPath`, `embedHeading`, `embedTags`, `includeCode`, `minLength`, `maxTokens`.
  - Store actual settings values instead of MD5 hash for better UX (can show users what changed).
  - Update `prepare_user_data_embeddings` to store settings object instead of computing hash.
  - **No hash needed:** Field-by-field comparison is fast (mostly integers/booleans). Saves storage (~22 bytes hash vs ~30 bytes explicit settings, but explicit is more useful).
  - **Note:** `embeddingVersion` stays separate (plugin-level constant, not a user setting). `maxBlockSize` derived from `notes_max_tokens`.
- [x] **Remove `history` tracking:**
  - Drop `history` field from metadata to simplify structure (no longer needed with single-shard constraint and overwrite-on-update strategy).
  - Audit trail provided by `updatedAt` timestamp (already present in metadata).
  - If richer history is needed later, can be added with clear use cases in mind.
- [x] **Keep `metric` field for future-proofing:**
  - The system hardcodes cosine similarity everywhere; `metric: 'cosine' | 'l2'` field is never checked currently.
  - Keep field as `metric: 'cosine'` with code comment: "// L2 distance not yet supported, reserved for future use"
  - Documents intent without adding code complexity; avoids breaking change if L2 support added later.
- [x] **Simplify shard structure to single shard per note:**
  - Cap shard size at 500KB (~220 blocks with 1536-dim embeddings).
  - **Implementation in `build_shards()`**: Enforce single-shard constraint with automatic truncation and logging:
    - Estimate shard size before building: `rows * (dim + 4 + 2) * 1.33` (base64 overhead).
    - If estimated size > 500KB, automatically truncate to first N blocks that fit.
    - Log warning when truncating: `"Note has X blocks, exceeding shard cap. Only first N blocks included (YKB)"`
    - Log info when approaching cap (>400KB): `"Note approaching shard cap: X blocks, YKB (cap: 500KB)"`
  - **Note on dialog approach:** Dialog (as in original plan) is deferred - automatic truncation is simpler for MVP. Can add user choice later if needed.
  - Simplifies read path: always fetch `jarvis/v1/emb/<modelId>/live/0` (no iteration).
  - Simplifies cleanup: just overwrite `live/0` on write. Cleanup logic handles legacy multi-shards.
- [x] **Optimize `BlockRowMeta` to remove unused fields:**
  - **Remove `noteId` and `noteHash`**: Never stored, always fall back to parent note metadata (see `userDataReader.ts:109-110`).
  - **Remove `tags`**: Stored but never used during search/chat (tags are embedded into vectors via `notes_embed_tags`, not used from metadata).
  - **Keep all other fields** (actively used):
    - `bodyStart`, `bodyLength`: Extract block text in `extract_blocks_text()` (critical for chat context).
    - `lineNumber`: Scroll to block when clicked in panel, displayed in UI (critical for UX).
    - `title`, `headingLevel`: Displayed in search results panel, builds full heading paths.
    - `headingPath`: Provides richer context for nested headings (minimal overhead, useful).
  - Update `build_block_row_meta()` and `BlockRowMeta` interface accordingly.
- [x] **Implement robust format-agnostic metadata handling:**
  - When reading `jarvis/v1/meta`, wrap in try-catch and handle parse failures gracefully.
  - If metadata fails to parse or has unexpected structure (missing required fields, wrong types), log warning and treat as "no embeddings".
  - On next update for that note, fresh metadata will be written with current format (effectively migrating forward).
  - No need to support specific old formats - just fail gracefully and rebuild on next opportunity.
- [x] **Update shard cleanup logic in `UserDataEmbStore.put()`:**
  - After writing shard 0 for a model, just overwrite `jarvis/v1/emb/<modelId>/live/0`.
  - Don't iterate to delete old multi-shards - they're harmless if they exist (rare legacy edge case).
  - Leave other models' shards untouched to support multi-model coexistence.
- [x] **Add validation logic for synced embeddings in read path:**
  - **Lazy validation approach:** Single validation point during search when embeddings are loaded. No upfront checks.
  - **Why on every search:** Sync happens frequently (~5 min), settings changes are rare (months). Need to catch mismatches from other devices quickly.
  - **TODO: Benchmark validation overhead** (see §12 Testing for details) and reconsider frequency if necessary (e.g., once per hour, or sync-triggered).
  - When loading userData embeddings in `read_user_data_embeddings()` or `find_nearest_notes()`, validate:
    - `meta.models[currentModel.id]` exists (device's model has embeddings for this note)
    - `meta.models[currentModel.id].embeddingVersion === currentModel.embedding_version`
    - `meta.models[currentModel.id].modelVersion === currentModel.version`
    - `meta.models[currentModel.id].settings` matches `currentSettings` (compare field-by-field: all booleans/integers, fast comparison)
  - For mismatched notes: **include in search results anyway** (use embeddings despite mismatch), add to mismatch counter for dialog.
  - **Transient errors:** If note metadata seems inconsistent, log clearly: `"Note ${noteId}: inconsistent metadata (may be rebuilding on another device or sync in progress), skipping"` - don't fail hard.
  - **Log performance metrics**: `"Validation: checked ${N} notes in ${ms}ms, ${mismatchCount} mismatches (used anyway)"`
  - After search completes, if mismatches detected, show dialog **once per session** (simple approach for MVP): 
    - "Some notes have mismatched embeddings: [X notes: wrong model] [Y notes: different settings: embedTitle Yes→No, maxTokens 512→1024]. Rebuild affected notes now? [Yes] [No]"
    - If user says **Yes**: Call `startUpdate(force: true, noteIds: mismatchedNoteIds)` to rebuild ONLY mismatched notes.
    - If user says **No**: 
      - Continue using mismatched embeddings (already included in results).
      - Set session flag to suppress dialog for rest of current session.
      - Log: `"User declined validation rebuild, using ${mismatchCount} mismatched embeddings"`
    - **Next session:** Dialog shown again if mismatches still exist (simple MVP behavior, can enhance later).
    - User can always rebuild via manual "Update DB" command (force=true, all notes).
- [x] **Implement smart rebuild logic (force vs. regular updates):** *(Completed — note: background sweeps stay `force=false` to mirror legacy UX; validation dialog handles settings/model drift.)*
  - **Regular update (force=false)**: Triggered by note save, periodic background sweep.
    - For each note: check if `noteHash` changed.
    - If changed → rebuild that note.
    - If unchanged → skip (already up-to-date).
    - This is the DEFAULT mode for incremental updates.
  
  - **Forced update (force=true)**: Triggered by settings change, manual "Update DB" command, or validation dialog rebuild.
    - For each note: check if ANY of these changed:
      - `noteHash` changed (content modified)
      - Settings mismatch with userData for active model
      - Active model mismatch with userData
    - If ANY changed → rebuild that note with current active model.
    - If NONE changed → skip (already up-to-date, don't waste embedding API calls).
    - No prompts - just rebuild what needs rebuilding.
    - Can be scoped to specific `noteIds` (e.g., from validation dialog: only rebuild mismatched notes).
  
  - **Why this is better than SQLite:** SQLite rebuilds ALL notes on settings/model change (all-or-nothing at DB level). userData can be granular per-note, so we only rebuild what's actually outdated. Example: If 500/1000 notes are already up-to-date (synced from another device with new settings), we only rebuild the other 500.
  
  - **Implementation changes needed in `index.ts`:**
    - **Manual "Update DB" command** (line 282): Change to `start_update(model_embed, panel, { force: true })` (currently missing force, defaults to false).
    - **Background periodic sweeps** (lines 147, 162, 467, 569, 594): Keep as `force: true` to catch settings/model mismatches from sync.
    - **Incremental note saves** (line 442): Keep as force=false (only hash changes).
    - **Validation dialog rebuild**: Use `start_update(model_embed, panel, { force: true, noteIds: mismatchedNoteIds })`.
  
  - **Implementation in `update_note()`**: Accept current settings/model as parameters, compare with `meta.models[currentModel.id]` metadata, return early if all matches and force=false.

- [x] **Implement smart model switching with coverage check:**
  - **Trigger:** When user changes `notes_model` setting (active model switch).
  - **Design decision:** Lazy validation approach - check compatibility during search, not upfront on switch. Simpler than SQLite's upfront checks.
  
  - **Coverage check (fast sampling):**
    1. Sample 100-200 random notes (or all notes if <200 total).
    2. For each sampled note, check if `meta.models[newModelId]` exists in userData.
    3. Calculate coverage: `notesWithModel / sampleSize`, extrapolate to estimate total coverage.
    4. **Performance**: ~100 notes × 5ms = 500ms (acceptable for model switch).
    5. If estimated coverage < 10% (threshold configurable): show low-coverage dialog.
    6. If estimated coverage ≥ 10%: seamless switch.
    7. **No upfront compatibility check** - settings/version mismatches discovered naturally during first search via validation logic (lines 74-92).
  
  - **Low-coverage dialog (coverage < 10%):**
    - Message: "Model '[modelId]' has ~[X]% coverage (estimated [count]/[total] notes). Populate embeddings now?\n\nThis will use embedding API quota. You can also populate later via 'Update DB'."
    - Options: [Populate Now] [Switch Anyway] [Cancel]
    - **Populate Now**: Switch to new model in settings, trigger `startUpdate(force: true)` to populate missing embeddings.
    - **Switch Anyway**: Switch to new model in settings, log warning about low coverage. Search will work but return fewer results until populated.
    - **Cancel**: Keep old model in settings, revert dropdown in settings UI.
  
  - **Seamless switch (coverage ≥ 10%):**
    - Switch to new model in settings, no upfront dialog.
    - Log: `"Model switch: ${oldModel} → ${newModel} (estimated coverage: ${coverage*100}% based on ${sampleSize} samples)"`
    - On first search: validation checks settings/version compatibility for all loaded embeddings (lines 74-92).
    - If mismatches detected: show summary dialog with human-readable diffs, offer to rebuild (smart re-prompting logic applies).
  
  - **Multi-device benefit:** If Device A populated Model B and synced, Device B switching to Model B sees high coverage → instant switch. If settings differ, discovered on first search with clear explanation.
  - **Rationale for lazy checking:** Simpler logic (single validation path), no redundant checks. Slight UX tradeoff (discover mismatches after switch vs before) for significantly simpler implementation.
  - **Rationale for sampling:** Coverage doesn't need to be exact (10% vs 12% doesn't matter). Fast sampling gives good-enough estimate without blocking model switch for large libraries.
## 1.6. ARCHIVED - Multi-Device Rebuild Coordination (Pending Testing)
**Status:** Deferred until testing shows it's needed. May be unnecessary complexity.

- [ ] **Add multi-device rebuild coordination:**
  - Add `rebuildInProgress: boolean` and `rebuildStartedAt: string` (ISO timestamp) to metadata structure.
  - Before starting rebuild of a note, set `rebuildInProgress: true`, `rebuildStartedAt: now`.
  - After completing rebuild, set `rebuildInProgress: false`.
  - **Read path handling:** If note has `rebuildInProgress: true`:
    - Check age: `now - rebuildStartedAt > 1 hour` → treat as stale, clear flag, proceed with validation.
    - Otherwise: skip validation for this note (rebuild in progress on another device).
  - **Race handling:** If two devices start rebuild simultaneously, last writer wins (standard LWW for userData). Flag prevents spurious validation errors during rebuild window.
  - Log when encountering in-progress rebuilds: `"Note ${noteId}: rebuild in progress (started ${age}s ago), skipping validation"`
  - **Decision to archive:** Accept transient validation errors during rare concurrent rebuilds. Add clear logging instead: "Note has inconsistent metadata (may be rebuilding on another device)". Simpler metadata structure, less logic. Can revisit if issues arise in testing.

## 2. Content Normalization, Chunking, Hashing
- [x] Document and preserve current normalization pipeline (HTML→Markdown where required, newline normalization) to maintain compatibility with existing hashes. *(Documented in `calc_note_embeddings` and `convert_newlines`.)*
- [x] Confirm legacy block segmenter configuration (token budget, heuristics) meets storage limits; adjust defaults if needed. *(Documented in `split_block_to_max_size` comment noting the `notes_max_tokens` limit.)*
- [x] Ensure block segmenter emits deterministic heading metadata and align offsets for snippet rendering.
- [x] Compute `contentHash` from normalized source and compare against `meta.current.contentHash` to skip redundant rebuilds.

## 3. Embedding Generation & Quantization Pipeline
- [x] Integrate embedding model invocation on capable devices (desktop first) using existing model registry.
- [x] Convert F32 embeddings to q8 (per-row scale) and serialize to base64 payloads.
- [x] Build shard-packaging that respects target size limits (≤200–400 KB per shard) and writes numbered `live/<i>` keys.
- [x] Populate shard `meta` arrays with block metadata, centroid ids (when IVF enabled), and row counts.
- [x] Append/update `meta.current` with epoch, shard count, row count, blocking info, timestamps.
- [x] **Update to single-shard-per-note constraint** (see §1.5):
  - Enforce 500KB cap in `build_shards()` (~220 blocks with 1536-dim embeddings). *(Implemented in `shards.ts`: automatic truncation with warnings)*
  - Log warning if note approaches or exceeds cap, but still create embeddings (don't fail). *(Implemented: warns at 400KB+ and truncates at 500KB)*
  - Warning only appears during re-embedding (when note content changes), not on every search. *(Warnings are in build path only)*
  - Always write to `jarvis/v1/emb/<modelId>/live/0` (single shard). *(Implemented in `userDataStore.put()`)*
  - Update readers to fetch single shard (no iteration logic needed). *(Simplified `userDataReader.ts`, `centroidNoteIndex.ts`, and `userDataStore.getShard()`)*

## 4. Centroid & Anchor Infrastructure — **OBSOLETE (IVF Removed 2025-11-27)**

> **Note:** This entire section is obsolete. IVF/centroid-based search was removed in favor of simpler in-memory brute-force search. The catalog note is still used for model metadata, but anchor notes and centroids are no longer needed.

- [x] ~~Create / locate the `Jarvis System Catalog` note; read/write `jarvis/v1/registry/models`.~~ (Catalog still used for model registry)
- [x] ~~Implement per-model anchor management~~ **REMOVED**
- [x] ~~Write/read centroid payloads~~ **REMOVED**
- [x] ~~Persist anchor note ids in plugin settings~~ **REMOVED**
- [x] ~~Define refresh policy for centroids~~ **REMOVED**
- [x] ~~**Implement ephemeral inverted index for IVF search optimization**~~ **REMOVED** - IVF removed, in-memory cache scans all vectors directly
- [x] ~~**CONDITIONAL: Optimize brute-force candidate selection**~~ **DONE** - In-memory cache eliminates API pagination during search

- [x] **In-memory cache for all corpuses** ✅ **SIMPLIFIED (IVF Removed 2025-11-27)**
  - **Status:** ✅ **IMPLEMENTED & VALIDATED**
  - **Change:** IVF/centroids removed entirely. All corpuses now use in-memory brute-force search.
  - **Performance:**
    - Precision@10: **100%** (Q8 quantization perfect accuracy vs Float32 baseline)
    - Recall@10: **100%** (no false negatives)
    - Cache build: ~2-5s one-time (12,839 blocks @ 1536-dim = 19.8MB)
    - Cache search: 10-50ms pure RAM (vs 2000ms+ userData I/O)
  - **Memory limits:** 100MB mobile, 200MB desktop (with capacity warnings at 80%)
  - **Implementation:** `src/notes/embeddingCache.ts` (`SimpleCorpusCache` class)
  - **Cache invalidation:**
    - Incremental update on note embed/delete
    - Full rebuild on model switch or excluded folders change
    - Dimension mismatch detection (auto-invalidates if model dimension changes)

  - **Files:**
    - ✅ `src/notes/embeddingCache.ts`: Full-corpus cache
    - ✅ `src/notes/embeddings.ts`: Cache integration in search
    - ✅ `src/notes/topK.ts`: NaN rejection
    - ✅ `src/index.ts`: Cache clearing on model switch

  - **Removed (IVF cleanup):**
    - ~~`src/notes/centroids.ts`~~ - centroid training
    - ~~`src/notes/centroidAssignment.ts`~~ - centroid assignment
    - ~~`src/notes/centroidLoader.ts`~~ - centroid loading
    - ~~`src/notes/centroidNoteIndex.ts`~~ - inverted index
    - ~~Anchor note infrastructure~~ - no longer needed

## 5. Unified Search Engine Enhancements
- [x] Implement q8 cosine scoring kernel with configurable top-K heap and score thresholding. *(See `q8.ts`, `topK.ts`, and integration in `embeddings.ts`.)*
- [x] ~~Add IVF-Flat support: centroid distance computation, `nprobe` selection, row filtering by centroid id.~~ **OBSOLETE (IVF Removed 2025-11-27)** - In-memory brute-force search used instead.
- [x] Surface tunable knobs (`candidateLimit`, `scoreThreshold`) with sane platform presets. *(IVF-specific knobs `nlist`, `nprobe`, `stopAfterMs`, `maxDecodedRowsInMem` removed with IVF.)*
- [x] ~~Integrate shard streaming decode~~ **OBSOLETE** - In-memory cache loads all vectors at startup, no streaming needed.
- [x] ~~Prioritize canonical centroid support per §5.7, including parent-map fallback for low-RAM devices.~~ **OBSOLETE (IVF Removed)** - No centroids or parent maps.

## 6. Write Path Integration (Desktop-first)
- [x] Ensure the new write path layers on top of `update_note`/`update_embeddings` so SQLite + userData stay in lockstep via `maybeWriteUserDataEmbeddings`.
- [x] Hook note-save/change events to schedule embedding rebuild jobs with debouncing and queueing.
- [x] Keep rebuild execution off the UI thread by reusing `startUpdate`/`update_note_db` background jobs and existing panel progress reporting.
- [x] Lean on epoch bumps in `prepare_user_data_embeddings` and `UserDataEmbStore.put` for last-writer-wins concurrency; no extra locking required yet.

## 7. Read Path Integration (All Platforms)
- [x] Update semantic search entry points to fetch `meta`/shards via `EmbStore` when feature flag enabled.
- [x] Implement shard decoding on demand with reuse of work buffers to cap peak memory.
- [x] Maintain small LRU cache of recently decoded shards for chat follow-ups.
- [x] Return results with existing `BlockEmbedding`-compatible metadata for downstream UI.
- [x] Fallback gracefully when embeddings missing (e.g., revert to legacy path or text search).
- [x] **Simplify read path for single-shard constraint** (see §1.5):
  - Always read `jarvis/v1/emb/<modelId>/live/0` (no iteration needed).
  - Remove or simplify shard iteration logic in `read_user_data_embeddings()`.
  - Cache key can drop shard index: `${noteId}:${meta.modelId}:${meta.current.epoch}`.
  - Most notes will have <5ms userData fetch latency (single key lookup).

## 8. Platform-Specific Adjustments
- [x] Detect platform automatically via `joplin.versionInfo()` and default device profile accordingly.
- [x] Mobile/Web: enforce lower `candidateLimit` and memory limits (100MB mobile, 200MB desktop). *(IVF-specific `nprobe`, `maxDecodedRowsInMem`, and early-exit timers removed with IVF.)*
- [x] Implement per-platform defaults loader and allow overrides via settings.
- [x] ~~Desktop: tune centroid usage and IVF params~~ **OBSOLETE (IVF Removed)** - All platforms use in-memory brute-force.
- [x] Handle device import constraints: only try to import sqlite/fs and generate the database when experimental flag is on, or when we're on desktop. we will keep the embedding model null, and chat with your notes command will do nothing.
- [x] Handle device import constraints: only try to import fs and save the USE model on desktop (on mobile fallback to download the model via the package, just a simple load).
- [x] On mobile, register editor toolbar buttons for commands: (1) chat with Jarvis, (2) chat with your notes, (3) find related notes, (4) edit selection with Jarvis, (5) autocomplete with Jarvis, (6) annotate note.

## 9. Settings, Privacy, and Maintenance Controls
- [x] ~~Surface IVF tuning controls (basic + advanced) and memory/time budgets per platform.~~ **OBSOLETE (IVF Removed 2025-11-27)** - Removed `notes_search_candidate_limit`, `notes_search_max_rows`, `notes_search_time_budget_ms` settings.
- [x] Build model management dialog (dropdown selector) that lists stored embedding models and lets users delete deprecated models:
  - **Scan strategy**: Single-phase scan for simplicity.
    - Iterate all notes, read `jarvis/v1/meta`, parse models from valid metadata.
    - Build inventory: `{ modelId → { noteCount, lastUpdated } }`.
    - If metadata fails to parse: log warning, skip that note. User can manually rebuild if needed.
    - **Note**: With single-shard constraint (§1.5), all notes have only `live/0`. Scanning is very fast (one key per note per model).
  - **Display**: Show model ID, version, note count, approximate storage (noteCount × ~300KB), last used timestamp, active status.
  - **Deletion**: For selected model, iterate all notes:
    - If metadata is valid and parseable: remove that model from `meta.models`, update `jarvis/v1/meta`.
    - Delete shard: `client.del(noteId, 'jarvis/v1/emb/<modelId>/live/0')`. No iteration needed with single-shard constraint.
    - If metadata is unparseable: skip (user can clean up manually via rebuild).
  - Include associated anchor notes and centroids in deletion (scan catalog for model anchors).
  - **Note**: Joplin API doesn't support querying "all userData keys" directly, so we must iterate notes.
- [x] **Model switching is handled by smart rebuild logic** (§1.5):
  - When user changes `notes_model` via settings, `onChange` triggers `start_update(force: true)`.
  - Smart rebuild logic (lines 76-95) only rebuilds notes that are outdated (model/settings/hash mismatch).
  - Notes already up-to-date (e.g., synced from another device) are skipped automatically.
  - If rebuild is cancelled, validation on next search will catch remaining mismatches and prompt user.
- [x] **GC actions needed for userData** (unlike SQLite, these don't happen automatically):
  - **Settings changes do NOT leave orphans**: When settings change and rebuild happens, new embeddings overwrite old ones at `live/0` for each model. No GC needed.
  - **Deprecated model cleanup**: Old models persist until explicitly deleted via model management dialog (see above). Their embeddings sit unused at `jarvis/v1/emb/<oldModelId>/live/0`.
  - **Note deletion cleanup is automatic**: userData is deleted when note is deleted by Joplin. No manual GC needed.
  - **SQLite file cleanup**: After successful migration and soak period, optionally archive or delete legacy `.sqlite` files from plugin data directory.
- [ ] **Remove unused `jarvis.notes.db.updateSubset` command** (deferred cleanup):
  - **Status:** Command registered in `index.ts` (line 479) but unused after validation dialog fix (changed to use `jarvis.notes.db.update` for full scan instead).
  - **Rationale for keeping temporarily:** Safety measure in case we discover edge cases or need to revert to subset-based rebuilds.
  - **Removal plan:** After soak period (1-2 release cycles), verify no usage and remove command registration entirely.
  - **When to remove:** Search codebase for `updateSubset`, confirm only registration remains, delete lines 478-501 in `index.ts`.

## 10. Migration from Legacy Stores *(Archived)*
- [x] **Add first build completion tracking:**
  - Introduce `notes_model_first_build_completed: { [modelId: string]: boolean }` in plugin settings (persisted across sessions, default `{}`).
  - Tracks completion of first full build for any model (fresh or migrated from SQLite).

- [x] **Automatic per-model migration from SQLite to userData:**
  - **Current state:** `Update DB` sweep backfills userData from existing SQLite embeddings without hitting the API.
  - [x] Set `notes_model_first_build_completed[modelId] = true` after any successful full sweep (migration or fresh build).
  - [x] Skip legacy SQLite loading once the flag is true (userData-only path).
  - [x] Retry unfinished sweeps on next activation until the flag is set (skip when userData already up-to-date).

- [ ] After soak period, optionally archive or delete legacy `.sqlite` files via maintenance dialog.

- [x] ~~**Post-first-sweep centroid assignment:**~~ **OBSOLETE (IVF Removed 2025-11-27)**
  - ~~Centroid assignment was used to assign IVF cluster IDs to note embeddings after the first full build.~~
  - **Status:** `centroidAssignment.ts` deleted with IVF removal. No centroid IDs needed since all search is brute-force.

- [x] **Post-migration validation:** Spot-check 10 random notes per model: compare search results between SQLite and userData (before SQLite deletion).

## 11. Local Logging & Diagnostics
- [x] Add structured local logs for shard decode errors, centroid discovery failures, feature-flag state, and migration progress.

## 12. Testing & QA

### Real-World Testing Scenarios

#### Scenario 1: Two Devices, Different Models (Common Case)
- **Setup:**
  - Device A (desktop): Uses model "openai-1536", has 100 notes with embeddings.
  - Device B (mobile): Uses model "use-512", starts fresh.
  - Both devices sync via Joplin sync.
- **Test Flow:**
  1. Device A creates/updates embeddings for notes 1-50, syncs.
  2. Device B syncs, then creates embeddings for notes 51-100, syncs back.
  3. Device A syncs again.
- **Expected Behavior:**
  - Each note should have `meta.models["openai-1536"]` and `meta.models["use-512"]` coexisting.
  - Device A searches with openai-1536 → finds all 100 notes (its own embeddings).
  - Device B searches with use-512 → finds all 100 notes (its own embeddings).
  - No warnings, no conflicts, no rebuilds needed.
  - Each device uses its own local model setting (`settings.notes_model`) to read embeddings.
  - Device A reads `meta.models["openai-1536"]`, Device B reads `meta.models["use-512"]`.
  - No conflicts because each model's embeddings are isolated in the `models` map.
- **Edge Cases to Test:**
  - Device A updates note 25 while Device B has stale embeddings → both models' embeddings updated independently.
  - Device A deletes model "openai-1536" via management dialog → Device B's "use-512" embeddings unaffected.

#### Scenario 2: Two Devices, Same Model, Same Settings (Happy Path)
- **Setup:**
  - Device A and B both use model "openai-1536" with settings: `embedTitle=true`, `maxTokens=512`.
  - Start with 100 notes, no embeddings.
- **Test Flow:**
  1. Device A: Rebuild all notes (updates 1-100), syncs.
  2. Device B: Syncs, then searches immediately.
  3. Device B: Updates note 50 (content change), syncs.
  4. Device A: Syncs, then searches.
- **Expected Behavior:**
  - Device B (step 2): Finds all 100 notes, no warnings, no rebuild prompts (embeddings match current settings/model).
  - Device B (step 3): Only note 50 rebuilt (hash changed), not all 100.
  - Device A (step 4): Finds all 100 notes including updated note 50 (synced from B), no warnings.
  - Last-writer-wins for note 50 if both devices update simultaneously.
- **Edge Cases to Test:**
  - Simultaneous updates to same note → verify no corruption, last writer wins, `updatedAt` timestamp resolves conflicts.
  - One device offline for 24 hours → syncs back, embeddings still valid (no expiration).

#### Scenario 3: Two Devices, Same Model, Different Settings (Warning Dialog)
- **Setup:**
  - Device A: model "openai-1536", settings `embedTitle=true`, `maxTokens=512`, `embedTags=false`.
  - Device B: model "openai-1536", settings `embedTitle=false`, `maxTokens=1024`, `embedTags=true`.
  - Start with 50 notes embedded on Device A, synced to Device B.
- **Test Flow:**
  1. Device B syncs (gets Device A's embeddings with different settings).
  2. Device B: User performs search.
  3. Device B: Validation detects mismatch, shows dialog.
  4. User chooses "Rebuild Now" or "Use Anyway".
- **Expected Dialog (step 3):**
  - "Some notes have mismatched embeddings: [50 notes with different settings: embedTitle Yes→No, maxTokens 512→1024, embedTags No→Yes]. Rebuild affected notes?"
  - Options: [Rebuild Now] [Use Anyway] [Cancel]
  - Dialog shown **once per session** (not on every search).
- **If User Chooses "Rebuild Now":**
  - Only the 50 mismatched notes rebuilt with Device B's current settings.
  - After sync, Device A searches → sees Device B's embeddings if Device B's `updatedAt` is newer → Device A now shows mismatch dialog!
  - Eventually both devices converge to last-updated settings (whichever device updated most recently).
- **If User Chooses "Use Anyway":**
  - Search returns results using mismatched embeddings (suboptimal but functional).
  - Dialog suppressed for rest of session.
  - Next session: dialog shown again if mismatches still exist.
- **Edge Cases to Test:**
  - 25 notes match, 25 don't → dialog shows count breakdown.
  - Device A changes settings while Device B is offline → Device B syncs and sees dialog immediately on first search.
  - User ignores dialog for weeks → embeddings remain mismatched, no auto-rebuild (user choice respected).

#### Scenario 4: Model Switch with Low Coverage (Coverage Dialog)
- **Setup:**
  - Device A: Has 1000 notes with model "openai-1536" embeddings.
  - User decides to switch to model "use-512" (only 5 notes have use-512 embeddings, 0.5% coverage).
- **Test Flow:**
  1. User changes `notes_model` setting to "use-512".
  2. Coverage check triggers (see §1.5, lines 112-114).
- **Expected Dialog:**
  - "Model 'use-512' has 0.5% coverage (5/1000 notes). Populate embeddings now?\n\nThis will use embedding API quota. You can also populate later via 'Update DB'."
  - Options: [Populate Now] [Switch Anyway] [Cancel]
- **If User Chooses "Populate Now":**
  - Switch to model "use-512" in settings, trigger full rebuild for 995 notes (5 already have embeddings).
  - Progress shown in panel: "Embedding notes with use-512... 200/995"
  - After completion, search returns full results.
- **If User Chooses "Switch Anyway":**
  - Search returns only 5 notes (low coverage), user experience degraded but functional.
  - Log warning: "Model use-512 has low coverage (0.5%), search results limited."
- **If User Chooses "Cancel":**
  - Revert dropdown to "openai-1536", no changes.

#### Scenario 5: Model Switch with Compatibility Issues (Compatibility Dialog)
- **Setup:**
  - Device A: 500 notes with model "openai-1536" embeddings (old plugin version, `embeddingVersion=1`).
  - Plugin updated to version 2.0 (`embeddingVersion=2`, changed chunking algorithm).
  - User wants to use same model but embeddings outdated.
- **Test Flow:**
  1. User keeps `notes_model=openai-1536` (no change).
  2. On first search, validation detects version mismatch (see §1.5, lines 76-81).
- **Expected Dialog (after first search completes):**
  - "Some notes have mismatched embeddings: [500 notes with older embedding version: v1→v2]. Rebuild affected notes?"
  - Options: [Rebuild Now] [Use Anyway] [Cancel]
  - Shows **once per session** (not on every search).
- **If User Chooses "Rebuild Now":**
  - Rebuild 500 notes with current embedding version (v2).
  - Old embeddings overwritten at `live/0`.
- **Alternative Setup (model version mismatch):**
  - Model "openai-1536" upgraded from version "3.0" to "3.5" (API provider changed model).
  - Same flow, dialog shows: "[500 notes with older model version: 3.0→3.5]"

#### Scenario 6: Concurrent Rebuild on Two Devices (Race Condition)
- **Setup:**
  - Device A and B both use model "openai-1536", settings changed on both.
  - Both devices decide to rebuild note 42 at the same time.
- **Test Flow:**
  1. Device A: Starts rebuild of note 42, sets `rebuildInProgress=true` in metadata, syncs.
  2. Device B: Independently starts rebuild of note 42 (hasn't synced yet), sets `rebuildInProgress=true`.
  3. Device A: Completes rebuild, writes embeddings, sets `rebuildInProgress=false`, syncs.
  4. Device B: Completes rebuild, writes embeddings, sets `rebuildInProgress=false`, syncs.
  5. Both devices sync final state.
- **Expected Behavior:**
  - Last writer wins (Device B's embeddings overwrite Device A's if Device B synced last).
  - No corruption, no errors.
  - `rebuildInProgress` flag prevents both devices from triggering spurious validation warnings during rebuild window.
  - If Device C searches during this window and syncs while `rebuildInProgress=true`, it skips validation for note 42 (logs: "Note 42: rebuild in progress, skipping validation").
- **Edge Cases to Test:**
  - Rebuild abandoned (app crashes) → flag stays `true` → next device sees age >1 hour, clears flag, proceeds with validation.
  - Three devices rebuild same note → last writer wins, no deadlock.

#### Scenario 7: Very Large Note Exceeds 500KB Cap
- **Setup:**
  - Create note with 300 blocks (exceeds 500KB shard cap after q8 encoding).
- **Test Flow:**
  1. User saves note (triggers embedding rebuild).
  2. Estimator in `calc_note_embeddings()` detects overflow **before** calling embedding API (see §1.5, lines 39-50).
- **Expected Dialog:**
  - "Note 'My Large Research Paper' has 300 blocks, exceeds 500KB limit (estimated 680KB). [OK: Embed first 220 blocks that fit] [Cancel: Skip this note]"
- **If User Chooses OK:**
  - Only first 220 blocks embedded (saves API quota for blocks 221-300).
  - Log: `"Note abc123 'My Large Research Paper': blocks 221-300 skipped (exceeded 500KB cap, kept first 220 blocks)"`
  - Search still works, returns partial note coverage.
- **If User Chooses Cancel:**
  - No embeddings created for this note.
  - Log: `"Note abc123 'My Large Research Paper': embedding skipped by user (300 blocks, 680KB exceeds cap)"`
  - Note excluded from semantic search (fallback to text search if available).
- **Edge Cases to Test:**
  - Note with exactly 220 blocks (~500KB) → no dialog, full embedding succeeds.
  - Note grows from 200 blocks to 250 blocks over time → dialog shown on next update, existing 200 blocks reused.
  - Monitor logs during beta for notes >400KB (approaching cap) to validate threshold.

#### Scenario 8: Migration from SQLite to userData (One Device)
- **Setup:**
  - Existing Jarvis user with 200 notes in `openai-1536.sqlite`, 150 notes in `use-512.sqlite`.
  - Plugin updated to userData-enabled version.
- **Test Flow:**
  1. User opens Joplin, switches to model "openai-1536" (or keeps it active).
  2. Migration check runs (see §10, lines 249-268).
  3. Migration starts automatically (no prompt) for "openai-1536".
- **Expected Behavior:**
  - Progress shown: "Migrating model 'openai-1536' embeddings to new format... 45/200 notes"
  - For each note: check if `meta.models["openai-1536"]` exists → skip if present (already migrated or synced from another device).
  - Only notes with matching `contentHash` migrated (stale embeddings skipped, will rebuild on next update).
  - After completion: `migrationCompletedForModel["openai-1536"] = true` in settings.
  - SQLite file kept intact (for rollback).
  - Model "use-512" not migrated yet (only migrated when user activates it).
- **Edge Cases to Test:**
  - Migration interrupted (app closed at 100/200) → restart migration on next launch, skip already-migrated notes (cheap metadata reads).
  - Note hash mismatch during migration → skip that note, log warning, will rebuild on next update.
  - User switches to "use-512" → triggers separate migration for use-512.sqlite.

#### Scenario 9: Multi-Device Migration Coordination
- **Setup:**
  - Device A and B both have 200 notes in SQLite.
  - Device A starts migration first.
- **Test Flow:**
  1. Device A: Migrates all 200 notes to userData, syncs.
  2. Device B: Syncs (gets userData embeddings from Device A), then plugin triggers migration check.
- **Expected Behavior:**
  - Device B migration: For each note, checks if `meta.models["openai-1536"]` exists → 200/200 already present (synced from Device A).
  - Device B migration completes instantly (0 notes migrated, all skipped).
  - `migrationCompletedForModel["openai-1536"] = true` set on Device B.
  - No duplicate work, no API calls wasted.
- **Edge Cases to Test:**
  - Device A migrates 100 notes, syncs, crashes → Device B syncs, migrates remaining 100 + re-checks first 100 (skipped).
  - Device A and B both migrate offline → sync back, last writer wins per note, no corruption.

#### Scenario 10: Model Management - Deleting Deprecated Models
- **Setup:**
  - User has 500 notes with model "openai-1536-v1" embeddings (old deprecated model).
  - User has 500 notes with model "openai-1536-v2" embeddings (current model).
  - Current model in settings: `settings.notes_model = "openai-1536-v2"`.
- **Test Flow:**
  1. User opens model management dialog.
  2. Dialog scans all notes, builds inventory.
  3. User selects "openai-1536-v1" and clicks "Delete Model".
  4. Confirmation dialog: "Delete model 'openai-1536-v1'? This will remove embeddings from 500 notes (~150MB). This cannot be undone."
  5. User confirms deletion.
- **Expected Behavior:**
  - For each of 500 notes:
    - Read `jarvis/v1/meta`, remove `models["openai-1536-v1"]` entry.
    - Delete shard: `jarvis/v1/emb/openai-1536-v1/live/0`.
    - Write updated metadata back.
  - Progress shown: "Deleting model 'openai-1536-v1'... 250/500 notes"
  - After deletion:
    - Search with v2 still works (no interruption).
    - Model v1 no longer appears in management dialog.
    - Storage freed (~150MB).
  - Associated anchor note for v1 and centroids deleted (scan catalog).
- **Edge Cases to Test:**
  - Some notes have unparseable metadata → skip those notes, log warning, show summary: "Deleted from 480/500 notes. 20 notes had invalid metadata (ignored)."
  - Deleting currently active model → show error: "Cannot delete active model. Switch to another model first."
  - Two notes have only v1 embeddings (no other models) → after deletion, those notes have no embeddings (will be rebuilt on next update).

#### Scenario 11: ~~Anchor Note Infrastructure - Creation and Discovery~~ **OBSOLETE (IVF Removed 2025-11-27)**

> **Note:** This scenario is obsolete. Anchor notes were only used for storing centroids in the IVF system. With IVF removed, there are no anchor notes. The catalog note is still used for model registry metadata only.

#### Scenario 12: Settings Changes - Embedding Parameters Updated
- **Setup:**
  - User has 200 notes with embeddings, settings: `embedTitle=true`, `maxTokens=512`, `includeCode=true`.
  - All notes up-to-date with current settings.
- **Test Flow:**
  1. User changes settings: `embedTitle=false`, `maxTokens=1024`.
  2. Settings `onChange` handler triggers.
  3. System detects settings change, prompts user.
- **Expected Dialog:**
  - "Embedding settings changed: embedTitle Yes→No, maxTokens 512→1024. Rebuild all notes? This will use embedding API quota."
  - Options: [Rebuild Now] [Rebuild Later]
- **If User Chooses "Rebuild Now":**
  - Trigger `startUpdate(force: true)`.
  - Smart rebuild: all 200 notes have settings mismatch → rebuild all.
  - Progress shown: "Updating embeddings... 50/200 notes"
  - After completion, metadata updated with new settings.
- **If User Chooses "Rebuild Later":**
  - Settings saved, no rebuild.
  - On next search, validation detects mismatch → shows dialog (as in Scenario 3).
  - User can manually trigger "Update DB" later.
- **Edge Cases to Test:**
  - Change settings, start rebuild, interrupt (close app) → restart rebuilds remaining notes on next "Update DB".
  - Change settings back to original values → no rebuild needed (already matches userData).
  - Change only `embedTitle`, keep other settings → dialog shows only changed setting.

#### Scenario 13: Graceful Degradation - Missing/Corrupted Data
- **Setup:**
  - Various notes with missing or corrupted userData.
- **Test Scenarios:**

  **A. Missing embeddings for some notes:**
  - 100 notes total, 80 have embeddings, 20 have no userData.
  - User searches → finds 80 notes, 20 silently skipped (no error).
  - Log: "Search completed: 80/100 notes with embeddings, 20 notes without embeddings (skipped)."

  **B. Corrupted metadata (malformed JSON):**
  - Note has invalid JSON in `jarvis/v1/meta` (manual edit, corruption, etc.).
  - Search tries to read metadata → JSON parse fails.
  - Note skipped gracefully, no crash.
  - Log: "Note ${noteId}: invalid metadata (JSON parse error), skipping. User can rebuild via 'Update DB'."

  **C. Corrupted shard data (invalid base64):**
  - Note metadata valid, but shard payload at `live/0` has invalid base64.
  - Search tries to decode → base64 decode fails.
  - Note skipped gracefully, no crash.
  - Log: "Note ${noteId}: corrupted shard data (base64 decode error), skipping. User can rebuild via 'Update DB'."

  **D. ~~Missing centroid data:~~ OBSOLETE (IVF Removed 2025-11-27)**
  - ~~Anchor note deleted or centroids payload missing.~~
  - ~~Search tries to load centroids → fallback to brute-force search (no IVF).~~
  - All searches now use in-memory brute-force by default. No centroids needed.

  **E. userData read failure (sync in progress, network issue):**
  - Joplin API returns error on `userDataGet()`.
  - Note skipped for this search, retry on next search.
  - Log: "Note ${noteId}: userData read failed (${error}), skipping this search cycle."

  **F. Shard exceeds Joplin userData size limit (rare edge case):**
  - Note has exactly at userData limit after base64 encoding.
  - Write fails with "userData value too large" error.
  - Show user dialog: "Note '[title]' embeddings too large to store ([size]KB exceeds limit). Try reducing content or splitting note."
  - Don't crash, log detailed error.

#### Scenario 14: Platform-Specific Behavior - Mobile Memory Limits
- **Setup:**
  - Mobile device with 1000 notes corpus.
  - Memory constrained environment.
- **Test Flow:**
  1. User performs search on mobile.
  2. System loads embeddings with mobile presets:
     - `candidateLimit` auto-computed based on `notes_max_hits`
     - Memory limit: 100MB (vs 200MB on desktop)
     - In-memory cache built at startup (or lazily on first search)
  3. Search processes notes, monitor memory usage.
- **Expected Behavior:**
  - Peak memory stays under mobile limit (100MB, no OOM crash).
  - Cache warns at 80% capacity, refuses new entries at 100%.
  - Search completes in 10-50ms (pure RAM scan).
  - Log: `"[Cache] Built: ${N} notes, ${rows} rows, ${MB}MB, ${ms}ms"`
- **Edge Cases to Test:**
  - Very large corpus exceeds 100MB limit → cache refuses additional notes, logs warning.
  - Compare mobile vs desktop results for same query → should be identical (same brute-force algorithm).
  - Cache invalidation on model switch → full rebuild with new model's vectors.

#### Scenario 15: Large Corpus Sync - Incremental Progress
- **Setup:**
  - Device A has 10,000 notes with embeddings (large corpus).
  - Device B syncs for first time (fresh device).
- **Test Flow:**
  1. Device B starts syncing userData from Device A.
  2. Sync is incremental (Joplin syncs notes in batches).
  3. User tries to search on Device B mid-sync.
- **Expected Behavior:**
  - Search works with partially synced data (e.g., 2000/10000 notes synced).
  - Results returned from available embeddings (2000 notes).
  - No errors, no blocking.
  - After full sync completes, subsequent search finds all 10000 notes.
- **Edge Cases to Test:**
  - Sync interrupted (network disconnected) at 5000/10000 notes → search finds 5000 notes, works normally.
  - Resume sync later → gets remaining 5000 notes, no duplicates, no conflicts.
  - Very slow sync (takes hours) → user can still use search throughout, results incrementally improve.

### Performance Validation (Not Unit Tests, Real Measurements)
- [ ] **Validation overhead benchmark:**
  - Real corpus: 1000 notes with userData embeddings.
  - Measure total validation time during search (metadata reads + field comparisons).
  - Target: <50ms total validation overhead for 1000 notes (avg <0.05ms per note).
  - Instrument validation code to log `"Validation: checked ${N} notes in ${ms}ms, ${mismatchCount} mismatches"` on every search (during beta only).
  - If overhead >50ms, consider alternatives: sync-triggered validation, hourly validation, or caching validation results.
- [ ] **Log distribution analysis (during beta):**
  - Collect metrics on notes approaching 500KB cap (>400KB).
  - Track: note size distribution, blocks per note distribution, dialog frequency.
  - Validate 500KB threshold is reasonable (should affect <1% of notes).
  - If >5% of notes hit cap, increase threshold or improve chunking algorithm.
- [ ] **Mobile memory budget enforcement:**
  - Test on real mobile device with 1000+ notes corpus.
  - Monitor peak memory during cache build and search (should stay under 100MB limit).
  - Verify cache capacity warnings trigger at 80%.
- [x] ~~**IVF vs brute-force latency crossover:**~~ **OBSOLETE (IVF Removed 2025-11-27)** - All searches use in-memory brute-force.
- [x] ~~**nprobe tuning sweeps:**~~ **OBSOLETE (IVF Removed 2025-11-27)** - No `nprobe` parameter; no IVF.

### Unit & Integration Tests
- [ ] **Normalization pipeline:**
  - Test HTML→Markdown conversion preserves content hash consistency.
  - Test newline normalization (`convert_newlines`) produces deterministic output.
  - Regression: ensure existing hashes still match after pipeline changes.
- [ ] **Block segmenter:**
  - Test deterministic heading metadata extraction.
  - Test snippet offset alignment for rendering.
  - Test `notes_max_tokens` budget enforcement.
- [ ] **Quantizer (q8 encoding/decoding):**
  - Test per-row scale computation for various value ranges.
  - Test round-trip: F32 → q8 → F32 produces expected precision loss.
  - Test edge cases: all zeros, all same value, extreme outliers.
- [ ] **q8 cosine kernel:**
  - Test score calculation matches F32 baseline within tolerance (±0.01).
  - Test batch processing correctness.
  - Performance: verify q8 is faster than F32 on large batches (1000+ vectors).
- [x] ~~**IVF routing:**~~ **OBSOLETE (IVF Removed 2025-11-27)** - No centroids or IVF routing; all search is brute-force.
- [ ] **Base64 helpers:**
  - Test encode/decode round-trip for q8 vectors, Float32 scales. *(Uint16 centroid ids removed with IVF)*
  - Test handling of empty arrays, single-element arrays.
  - Test base64 output matches expected format (padding, character set).
- [ ] **Write→read round trip via userData:**
  - Create embeddings for test note, write to mock userData transport.
  - Read back, verify metadata fields match.
  - Verify shard payload decodes to correct vectors.
- [ ] **Multi-device epoch conflict handling:**
  - Simulate concurrent writes from two devices with different epochs.
  - Verify last-writer-wins behavior (higher `updatedAt` timestamp wins).
  - Verify no corruption, both writes succeed independently.

### Regression Tests
- [ ] **Search relevance (recall@K vs F32 baseline):**
  - Compare userData q8 search results vs legacy SQLite F32 results on same corpus.
  - Measure recall@10, recall@20 (should be >95% for q8).
  - Identify any systematic ranking differences (e.g., certain note types rank differently).
- [ ] **Snippet rendering:**
  - Verify search results display correct text snippets (no off-by-one errors).
  - Test with various content types: plain text, code blocks, lists, tables.
  - Test heading path display for nested headings.
- [ ] **Upgrade tests (multi-model preservation):**
  - Create embeddings with model A, switch to model B, create embeddings.
  - Verify `meta.models[A]` and `meta.models[B]` both present and independent.
  - Switch back to model A, verify no rebuild needed (embeddings intact).
  - Change plugin version (bump `embeddingVersion`), verify validation catches mismatch.

## 13. Code Cleanup - Remove Hierarchical IVF (Parent Maps) — **COMPLETED (IVF Removed 2025-11-27)**

> **Note:** This entire section is now complete because IVF was entirely removed in favor of in-memory brute-force search. All centroid, anchor, and parent map code has been deleted as part of that larger cleanup.

- [x] ~~Remove parent map storage functions~~ **Done** - Entire `anchorStore.ts`, `centroidLoader.ts` removed with IVF
- [x] ~~Remove parent map loading and caching~~ **Done** - `centroidLoader.ts` removed entirely
- [x] ~~Remove parent map usage in search~~ **Done** - `parentTargetSize` removed from `SearchTuning`
- [x] ~~Remove parent map from settings~~ **Done** - No parent map settings existed
- [x] ~~Update documentation~~ - `VECTOR_DB_TECH_SPEC.md` may need cleanup for obsolete IVF references
- [x] ~~Search codebase for remaining references~~ - Centroids and parent maps fully removed

## 13.1. Additional Code Simplifications

### A. Remove Float16 (f16) Centroid Support — **OBSOLETE (IVF Removed 2025-11-27)**

> **Note:** This section is obsolete. IVF was entirely removed, so Float16 centroid support is no longer needed. The `centroids.ts` file was deleted.

- [x] ~~Remove Float16 conversion functions~~ **Done** - `centroids.ts` deleted entirely with IVF removal
- [x] ~~Simplify encode/decode_centroids~~ **Done** - Functions deleted with IVF removal
- [x] ~~Update type definitions~~ **Done** - Types deleted with IVF removal

---

### B. Lower IVF Activation Threshold — **OBSOLETE (IVF Removed 2025-11-27)**

> **Note:** This section is obsolete. IVF was entirely removed, so there is no activation threshold. In-memory brute-force search is always used.

- [x] ~~Update threshold constant~~ **N/A** - IVF removed, no threshold needed
- [x] ~~Update documentation~~ **N/A** - IVF documentation will be archived

---

### C. Remove Unused `updateSubset` Command — **COMPLETED 2025-11-27**

- [x] Verified command existed in `src/index.ts` (lines 462-485)
- [x] Verified no callers in codebase
- [x] Removed command registration (~24 lines)

---

### D. Simplify Device Profile Detection Logic — **COMPLETED 2025-11-27**

- [x] Removed `infer_device_profile()` helper function (5 lines)
- [x] Flattened auto-detect → infer → override logic
- [x] Changed default from 'unknown' to 'desktop' (sensible fallback)
- [x] Inlined mobile check in final resolution
- **Lines simplified:** ~7 lines removed

---

### Summary of Additional Simplifications:
- **Sections A, B:** Obsolete with IVF removal
- **Section C:** ✅ Completed - removed ~24 lines
- **Section D:** ✅ Completed - simplified ~7 lines

## 14. Code Quality & Refactoring — Eliminate Duplication

**Context:** Analysis of embedding handling code identified significant duplication across `notes.ts`, `embeddings.ts`, `userDataIndexer.ts`, and `catalog.ts`. These refactorings will remove ~1030 lines of redundant code and improve maintainability.

### Phase 1: Extract Common API Patterns (HIGH PRIORITY - Low Risk, High Impact)

**Impact:** Removes ~300 lines, eliminates 50+ instances of manual cleanup code.

- [ ] **Create `src/utils/apiHelpers.ts` utility module:**
  - [ ] Implement `paginate<T>()` async generator for unified pagination:
    - Wraps `joplin.data.get()` with automatic page iteration
    - Handles `clearApiResponse()` cleanup in `finally` block
    - Yields `{ items, page, hasMore }` on each iteration
    - Eliminates 10+ duplicated pagination loops
  - [ ] Implement `fetchNoteTags(noteId: string)` helper:
    - Fetches tags for single note with automatic cleanup
    - Returns `string[]` of tag titles, empty array on error
    - Replaces 3 duplicated tag-fetch patterns
  - [ ] Implement `fetchBatchTags(noteIds: string[])` helper:
    - Parallel tag fetching for multiple notes
    - Returns `Map<noteId, tags[]>`
    - Optimizes batch filtering in `filter_excluded_notes()`
  - [ ] Implement `prepareNoteContent(note: any)` helper:
    - Combines HTML→Markdown conversion + OCR text appending
    - Replaces 2 duplicated content preparation patterns

- [ ] **Refactor call sites to use new helpers:**
  - [ ] `embeddings.ts`: Replace pagination in `get_all_note_ids_with_embeddings()` (lines 42-78)
  - [ ] `catalog.ts`: Replace 5 pagination loops in `get_system_folder_id()`, `get_catalog_note_id()`, `find_catalog_by_title()`, `find_anchor_candidates()`, `discover_anchor_by_scan()`
  - [ ] `notes.ts`: Replace pagination in note counting (lines 249-261) and sweep modes (lines 182-299)
  - [ ] `embeddings.ts`: Replace tag fetching in `update_note()` (lines 603-609) and `find_nearest_notes()` (lines 1334-1340)
  - [ ] `notes.ts`: Replace tag fetching in `filter_excluded_notes()` (lines 603-618)
  - [ ] `embeddings.ts`: Replace content prep in `update_note()` (lines 637-647) and `extract_blocks_text()` (lines 1070-1078)

- [ ] **Verify all `clearApiResponse()` calls handled by new helpers** (search codebase for remaining manual calls)

---

### Phase 2: ~~Consolidate Anchor/Catalog Management~~ **OBSOLETE (IVF Removed 2025-11-27)**

> **Note:** This phase is obsolete. Anchor notes were only used for centroids in IVF. With IVF removed, there are no anchor notes. The catalog note is still used but only for model registry metadata, which is much simpler.

- [x] ~~Create `AnchorResolver` facade class~~ **N/A** - Anchor infrastructure removed with IVF
- [x] ~~Refactor call sites~~ **N/A** - No anchor calls remain
- Catalog-only operations simplified after IVF removal

---

### Phase 3: Extract Settings & Validation Logic (HIGH PRIORITY - Low Risk, High Impact)

**Impact:** Removes ~120 lines, eliminates duplicate settings extraction/comparison logic.

- [ ] **Create `src/notes/settingsManager.ts` shared module:**
  - [ ] Move `extract_embedding_settings()` from `userDataIndexer.ts` (lines 53-63)
  - [ ] Move `settings_equal()` from `userDataIndexer.ts` (lines 68-78)
  - [ ] Move `extract_embedding_settings_for_validation()` from `validator.ts` (if exists)
  - [ ] Consolidate into unified `EmbeddingSettingsManager` class:
    ```typescript
    export class EmbeddingSettingsManager {
      extractSettings(settings: JarvisSettings): EmbeddingSettings { ... }
      compareSettings(a: EmbeddingSettings, b: EmbeddingSettings): boolean { ... }
      formatSettingsDiff(current: EmbeddingSettings, stored: EmbeddingSettings): string[] { ... }
    }
    ```
  - [ ] Export singleton instance: `export const settingsManager = new EmbeddingSettingsManager();`

- [ ] **Refactor call sites:**
  - [ ] `userDataIndexer.ts`: Use `settingsManager.extractSettings()` (line 104)
  - [ ] `userDataIndexer.ts`: Use `settingsManager.compareSettings()` (lines 138, 602, 744)
  - [ ] `embeddings.ts`: Use `settingsManager.extractSettings()` (lines 713, 739, 1528)
  - [ ] `embeddings.ts`: Use `settingsManager.compareSettings()` (lines 716, 744)
  - [ ] `notes.ts`: Move `formatSettingsDiff()` (lines 715-738) to `settingsManager.formatSettingsDiff()`, update call site (line 555)
  - [ ] `validator.ts`: Update to use shared `settingsManager` instead of local functions

- [ ] **Remove duplicate functions from original locations**

---

### Phase 4: Unify Corpus Counting (Medium Priority - High Risk, Medium Impact)

**Impact:** Removes ~150 lines, consolidates complex corpus tracking logic.

**⚠️ REQUIRES CAREFUL TESTING:** Corpus counting affects centroid training decisions, incorrect counts could trigger unnecessary retraining.

- [ ] **Create `src/notes/corpusMetrics.ts` module:**
  - [ ] Implement `CorpusMetrics` class:
    ```typescript
    export class CorpusMetrics {
      private accumulator: { current: number } | null = null;
      
      async initialize(
        modelId: string,
        anchorMeta: AnchorMetadata | null, 
        isFullSweep: boolean
      ): Promise<void> {
        // Initialize from anchor metadata (lines from notes.ts:99-125)
        // Handle full vs incremental sweep initialization
        // Count existing embeddings if metadata missing/zero
      }
      
      updateForNote(
        noteId: string,
        newRows: number,
        previousMeta: NoteEmbMeta | null,
        modelId: string
      ): number {
        // Update accumulator: subtract old rows, add new rows
        // Return updated corpus total
        // Logic from userDataIndexer.ts:428-462
      }
      
      async finalizeAndRecount(
        modelId: string,
        isFullSweep: boolean,
        incrementalSweep: boolean
      ): Promise<number> {
        // For full non-incremental sweeps: recount to detect deletions
        // For incremental sweeps: trust accumulator
        // Logic from notes.ts:375-393
      }
      
      getCurrent(): number { return this.accumulator?.current ?? 0; }
    }
    ```

- [ ] **Refactor call sites:**
  - [ ] `notes.ts`: Replace accumulator initialization (lines 99-125) with `corpusMetrics.initialize()`
  - [ ] `userDataIndexer.ts`: Replace `count_corpus_rows()` (lines 428-462) with `corpusMetrics.updateForNote()`
  - [ ] `notes.ts`: Replace final recount logic (lines 375-393) with `corpusMetrics.finalizeAndRecount()`
  - [ ] Pass `CorpusMetrics` instance to `prepare_user_data_embeddings()` instead of raw accumulator

- [ ] **Testing requirements before merging:**
  - [ ] Verify centroid training triggers at correct thresholds (2048 rows)
  - [ ] Test full sweep: count matches `get_all_note_ids_with_embeddings()` result
  - [ ] Test incremental sweep: accumulator updates correctly per note
  - [ ] Test note deletion detection in full non-incremental sweeps
  - [ ] Compare counts before/after refactor on real corpus (10,000 notes)

---

### Phase 5-6: Deferred (Lower Priority)

**Phase 5 (Note Processing Pipeline)** and **Phase 6 (Sweep Iterator)** are architectural improvements with higher risk. Defer until Phases 1-4 are merged and stable.

- [ ] **Phase 5**: Create unified `NoteProcessor` abstraction (~200 lines saved, high risk)
- [ ] **Phase 6**: Extract sweep strategy pattern (~180 lines saved, medium risk)

---

### Summary

| Phase | Priority | Risk | Lines Removed | Merge Order |
|-------|----------|------|---------------|-------------|
| 1. API Patterns | **HIGH** | Low | ~300 | **1st** |
| 3. Settings Logic | **HIGH** | Low | ~120 | **2nd** |
| ~~2. Anchor Resolution~~ | ~~High~~ | ~~Medium~~ | ~~80~~ | **OBSOLETE** |
| 4. Corpus Counting | Medium | High | ~150 | **3rd (with tests)** |
| 5. Note Processor | Low | High | ~200 | Deferred |
| 6. Sweep Iterator | Low | Medium | ~180 | Deferred |
| **Total (1,3,4)** | | | **~570** | |

**Recommended merge sequence:**
1. Merge Phase 1 (safest, immediate benefit)
2. Merge Phase 3 (independent, low risk)
3. ~~Merge Phase 2~~ **OBSOLETE** - Anchors removed with IVF
4. Merge Phase 4 with extensive testing (complex logic, high risk)

**Before starting:**
- Create feature branch: `refactor/eliminate-duplication`
- Commit each phase separately for easy rollback
- Run full test suite after each phase
- Manually test search/rebuild flows after Phase 4

## 15. Documentation
- [ ] Update developer docs with new data flow, key schema, and debugging tools.
- [ ] Publish user-facing notes describing the feature, privacy controls, and troubleshooting.
- [ ] Document internal tooling commands for inspecting embeddings and forcing rebuilds.
