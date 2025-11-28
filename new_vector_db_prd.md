# Jarvis: Per‑Note UserData Embeddings — Design Doc (Mobile‑safe)

## 1) Context & Problem Statement

Jarvis performs semantic search and retrieval‑augmented actions over Joplin notes. On desktop, we can persist embeddings/indexes via IndexedDB or filesystem. On **mobile and web**, panels/iframes have restricted storage (no IndexedDB/OPFS, no filesystem), and we want to **avoid storing large binary data in note bodies** (Markdown) to prevent editor bloat, accidental edits, and unnecessary full‑text indexing.

Additionally, a **central synced index** (single big artifact updated by multiple devices) is prone to sync conflicts. We want a design that:

* Works with **no file storage on mobile/web**.
* Avoids Markdown payload bloat and accidental edits.
* Minimizes sync conflicts across devices.
* Is fast enough for common libraries on mobile.

---

## 2) Goals & Non‑Goals

### Goals

1. **Portable storage of embeddings** that works across Desktop, Mobile, and Web, with **no file storage on mobile**.
2. **No Markdown bloat** and hidden from user editing.
3. **Low conflict rate** under multi‑device sync.
4. **Reasonable query latency** on mobile via in‑memory search and lightweight ANN or pre‑filtering.
5. **Schema versioning** to allow model upgrades and migrations.

### Non‑Goals

* Building a heavy on‑device ANN structure on mobile that persists across sessions (mobile remains ephemeral).
* Guaranteeing identical ranking across devices (acceptable to differ within tolerance).
* Replacing existing desktop‑only local caches (those can continue to exist if useful).

---

## 3) High‑Level Approach

Store per‑note embeddings and small metadata in **Joplin user data** (key‑value records associated with the note). Keys are **versioned and content‑addressed** to avoid conflicts. The note body remains free of binary blobs; at most we add a tiny marker/tag for discovery.

* **Write path (any device that can compute):** When a note's content (normalized) changes, recompute embeddings, quantize to **q8** (Int8), and write a single shard (vectors + per-row metadata) to **userData** keys under a versioned namespace. **Single-shard constraint:** Max 500KB per note (~220 blocks with 1536-dim embeddings); notes exceeding this are automatically truncated with warnings. Update a small per‑note meta key and set a tag (e.g., `jarvis-database`).
* **Read path (mobile/web):** On query, list candidate notes (by scope/tag/recentness), lazily fetch the userData shard for those notes, decode to memory, and run in‑memory vector search (IVF‑Flat q8 or brute‑force over a small shortlist). No local persistence is required. **Validation on every search:** Check that synced embeddings match current model/settings; prompt user to rebuild mismatched notes if needed.
* **Desktop enhancements:** Optionally build a fast local ANN (e.g., HNSW) for speed; not required for mobile.
* **Multi-model support:** Each note can store embeddings for multiple models simultaneously. An `activeModelId` tracks which model is used for queries. Switching models checks coverage and only rebuilds notes that need updating.

---

## 4) Architecture Overview

```
+--------------------+            postMessage            +----------------------+
|  Panel UI (all)    |  <----------------------------->  |  Plugin Main (all)   |
|  - chat, search    |                                  |  - compute policy    |
|  - filters/scope   |                                  |  - read/write userData|
+--------------------+                                  +-----------+----------+
                                                                    |
                                                                    | userDataGet/Set/Delete
                                                                    v
                                                         +----------+-----------+
                                                         |  Per-Note UserData   |
                                                         |  (hidden, synced)    |
                                                         +----------------------+
```

**Key choices**

* Storage: **per‑note userData** (not Markdown, not Resources) → invisible, synced, low conflict when content‑addressed or versioned.
* Discovery: **tag** (`jarvis-database` for catalog notes; exclusion tag `jarvis-exclude` with backward compatibility for `exclude.from.jarvis`) + cached registry (in settings) to avoid scanning all notes every time.
* Mobile: **ephemeral** (decode into memory per session); Desktop: optional persisted ANN cache.
* Multi-model: Each note supports multiple embedding models via nested metadata structure with `activeModelId` selector.

---

## 4.5) Unified Search Engine (desktop & mobile)

A single engine module (same code path on all platforms) with only **tunable knobs** per platform. Defaults aim for equivalence of results; desktop may just lift limits.

**Core algorithm (shared):**

1. Shortlist candidate notes (by scope, tags, or quick text search).
2. Load required **userData shards** lazily for those notes.
3. Compute query embedding; **q8 cosine** ranking with optional **IVF‑Flat** probing.
4. Return top‑K with per‑block provenance.

**Tunable knobs:**

* `candidateLimit` (e.g., desktop 2000, mobile 300–600)
* `ivf.lists` & `ivf.probeCount` (e.g., 512/24 desktop, 256/12 mobile)
* `stopAfterMs` soft budget; `scoreThreshold` for early exit

This ensures identical logic with different **thresholds/budgets**, not different algorithms.-----------------+            postMessage            +----------------------+
|  Panel UI (all)    |  <----------------------------->  |  Plugin Main (all)   |
|  - chat, search    |                                  |  - compute policy    |
|  - filters/scope   |                                  |  - read/write userData|
+--------------------+                                  +-----------+----------+
|
| userDataGet/Set
v
+----------+-----------+
|  Per-Note UserData   |
|  (hidden, synced)    |
+----------------------+

```

**Key choices**
- Storage: **per‑note userData** (not Markdown, not Resources) → invisible, synced, low conflict when content‑addressed.
- Discovery: optional **tag** (`jarvis.indexed`) + cached registry (in settings) to avoid scanning all notes every time.
- Mobile: **ephemeral** (decode into memory per session); Desktop: optional persisted ANN cache.

---

## 5) Data Model & Key Schema

### Namespacing
```

Namespace root: jarvis/v1

Per-note keys (embeddings):
- jarvis/v1/meta                                  // multi-model metadata + active model
- jarvis/v1/emb/<modelId>/live/0                  // single shard per model (500KB max)

Per-model anchor keys (centroids & maps):
- jarvis/v1/aux/centroids                          // Float32 (or Float16) centroids
- jarvis/v1/aux/parentMap/1024 | /512 | /256       // child→parent arrays
- jarvis/v1/aux/metadata                           // { modelId, nlist, dim, version, hash, … }

System Catalog (single note) keys:
- jarvis/v1/registry/models                        // { [modelId]: anchorNoteId }

````

### `aux` centroids value (JSON)

Stored on the **per-model anchor** under `jarvis/v1/aux/centroids`.

```json
{
  "format": "f32",
  "dim": 512,
  "nlist": 2048,
  "version": 1,
  "b64": "…base64 Float32Array of length nlist*dim…",
  "trainedOn": { "rows": 50000, "sampleStrategy": "reservoir" },
  "hash": "sha256:…",
  "updatedAt": "2025-10-12T02:30:00Z"
}
```

### `meta` value (JSON)
```json
{
  "activeModelId": "use-512",
  "models": {
    "use-512": {
      "dim": 512,
      "metric": "cosine",
      "modelVersion": "1.3.3",
      "embeddingVersion": 4,
      "settings": {
        "embedTitle": true,
        "embedPath": false,
        "embedHeading": true,
        "embedTags": true,
        "includeCode": false,
        "minLength": 50,
        "maxTokens": 512
      },
      "current": {
        "epoch": 7,
        "contentHash": "sha256:…",
        "rows": 240,
        "blocking": { "algo": "md-heuristic", "avgTokens": 512 },
        "updatedAt": "2025-10-12T02:30:00Z"
      }
    },
    "openai-1536": {
      "dim": 1536,
      "metric": "cosine",
      "modelVersion": "3.0",
      "embeddingVersion": 4,
      "settings": {
        "embedTitle": false,
        "embedPath": true,
        "embedHeading": true,
        "embedTags": false,
        "includeCode": true,
        "minLength": 100,
        "maxTokens": 1024
      },
      "current": {
        "epoch": 3,
        "contentHash": "sha256:…",
        "rows": 180,
        "blocking": { "algo": "md-heuristic", "avgTokens": 1024 },
        "updatedAt": "2025-09-15T14:20:00Z"
      }
    }
  }
}
````

**Key changes from original design:**
* **Multi-model structure:** `models` object allows multiple embedding models per note with independent metadata.
* **Active model selector:** `activeModelId` indicates which model is used for queries.
* **Explicit settings:** Instead of `settingsHash`, store actual `EmbeddingSettings` for better UX (users can see exactly what changed).
* **No history field:** Removed to simplify structure; `updatedAt` provides audit trail.
* **Single shard:** No `shards` count field since each model always has exactly one shard at `/live/0`.

`modelVersion`, `embeddingVersion`, and explicit `settings` mirror the metadata currently persisted in the sqlite `models` table; they allow every device to detect model or setting drift and trigger rebuild prompts when mismatches are found during search.

### `emb` shard value (JSON)

```json
{
  "epoch": 7,
  "format": "q8",
  "dim": 512,
  "rows": 100,
  "vectorsB64": "…base64 Int8Array…",
  "scalesB64": "…base64 Float32 per-row…",
  "centroidIdsB64": "…optional base64 Uint16Array…",
  "meta": [
    {
      "title": "Heading / Subheading",
      "headingLevel": 2,
      "headingPath": ["Note title", "Heading"],
      "bodyStart": 1234,
      "bodyLength": 456,
      "lineNumber": 78
    }
  ]
}
```

**Single-shard constraint:** Each model has exactly **one shard** at `jarvis/v1/emb/<modelId>/live/0`, capped at **500KB** (~220 blocks with 1536-dim embeddings). Notes exceeding this are automatically truncated during embedding with user-visible warnings. This simplifies read/write logic significantly.

**Optimized row metadata:** `BlockRowMeta` has been streamlined:
* **Removed fields:** `noteId`, `noteHash` (always inferred from parent note), `tags` (embedded in vectors, not needed at query time).
* **Kept fields:** `title`, `headingLevel`, `headingPath` (UI display), `bodyStart`, `bodyLength` (text extraction), `lineNumber` (scroll-to-block).

**Row metadata:** `meta` entries align 1:1 with the encoded vectors and carry everything today's `BlockEmbedding` consumers rely on for snippet rendering, jump-to-line, and UI display.

**Conflicts:** Writers use a new `epoch` for each build. Readers check `shard.epoch === meta.models[activeModelId].current.epoch` to avoid mixed generations. Meta is LWW; last complete writer wins.

---

## 5.5) Key enumeration constraints & GC strategy

* No key listing → avoid contentHash‑in‑key designs that create unlimited tombstones.
* **Single-shard simplification:**
  1. Writer picks `epoch = lastEpoch + 1` and writes shard to `live/0` with that epoch.
  2. Writer updates `meta.models[modelId].current` to `{ epoch, contentHash, rows, … }`.
  3. Legacy multi-shards (if any) are harmless and can be ignored—no cleanup iteration needed.
* **Cleanup:** When switching models or settings, new embeddings simply overwrite `live/0` for the affected model. Other models' shards remain untouched, supporting multi-model coexistence.
* **Model deletion:** Explicitly removing a model deletes `jarvis/v1/emb/<modelId>/live/0` and removes that model's entry from `meta.models`.

## 5.6) Centroids storage & discovery

**Two-tier anchor scheme (clear & human-friendly):**

* **System Catalog note** (single): title `Jarvis System Catalog`, tag `jarvis-database`. Its **content** explains the purpose and lists links to per-model anchors. It holds a small registry mapping `modelId → anchorNoteId` in userData: `jarvis/v1/registry/models`.
* **Per-model Anchor note(s):** one per `modelId`, title `Jarvis Model Anchor — <modelId> (v<version>)`, tags `jarvis-database`, `jarvis.model.<modelId>`. The **content** is user-friendly (what this note is, safe to keep, do not edit), while **centroids + maps** live in userData.

**Where centroids live (new):** On each **per-model anchor**, store centroids under `jarvis/v1/aux/centroids` (no modelId segment needed because the note itself encodes the model). Also store optional extras:

* `jarvis/v1/aux/parentMap/1024`, `/512`, `/256` (child→parent arrays)
* `jarvis/v1/aux/metadata` (e.g., `{ version, nlist, dim, hash, trainedOn, updatedAt }`)
* **`jarvis/v1/aux/inverted_index`** (NEW - CRITICAL FOR PERFORMANCE): Maps centroid IDs → note IDs containing blocks assigned to that centroid. Enables efficient note filtering before userData loading.

**Back-compat:** If only a single legacy anchor exists, we still read `jarvis/v1/aux/<modelId>/centroids` from it; during maintenance, migrate to per-model anchors and update the Catalog registry.

**Discovery flow:**

1. Read `jarvis/v1/registry/models` from the Catalog note's userData to get `anchorNoteId` for the desired `modelId`.
2. If missing, search notes tagged `jarvis-database` & `jarvis.model.<modelId>` and verify via userData key `jarvis/v1/aux/metadata.modelId`.
3. If still missing, create the per-model anchor idempotently; write centroids/maps when on a capable device; update the Catalog registry.

**Uniqueness & stability:**

* Persist `anchorNoteId` per `modelId` in settings for O(1) lookup; validate at boot by checking the per-model anchor’s `jarvis/v1/aux/metadata.modelId`.
* Even if users rename the note, discovery uses tag + userData, not title.
* All writes are to userData; note body stays human-readable guidance only.

### 5.7) Canonical centroids & parent maps (effective nlist)

* **Canonical pack:** Fix `nlist=2048` per `modelId` across all devices. Centroids are stored once on the per-model anchor.
* **Parent maps:** Store compact arrays mapping each child list to a parent for coarser granularities: `parentMap/1024`, `/512`, `/256`. A low-RAM device selects an **effective nlist** by probing parents, then expanding to their child lists.
* **Device knobs:** Devices vary only `nprobe`, candidate caps, and time budgets—**never** `nlist`.
* **Size note:** 2048×512×4 bytes ≈ 4.0 MB (Float32) → ~5.3 MB base64. Consider Float16 (`format: "f16"`) to halve size; store `format` in metadata and convert to F32 at runtime if needed.
* **Migration:** If older centroids exist on the legacy single anchor, copy them to the per-model anchor and write parent maps; update the Catalog registry.
* **Generation strategy:** Stream training samples from existing shards (reservoir sampling) and run a low-memory online k-means so even mobile can rebuild centroids in chunks; prefer reusing synced centroids when available and flag "pending rebuild" so a desktop can take over if a mobile run is too constrained.
* **Refresh policy:** Rebuild anchors when the embedding model version changes, centroid metadata hash drifts, or cumulative new rows exceed a threshold. Schedule recomputes opportunistically (idle desktop, device on charge) and mark notes so other devices know a refresh is in progress or needed.

### 5.8) Inverted Index for IVF Search Optimization (CRITICAL FOR PERFORMANCE)

**Problem:** Without an inverted index, IVF search must load metadata for **all N notes** in the database to check which blocks match the selected centroid IDs. This negates most of the IVF performance benefit, as metadata loading becomes the bottleneck (10,000 userData reads even if only 600 notes contain relevant blocks).

**Solution:** Maintain an **ephemeral in-memory** inverted index mapping `centroidId → Set<noteId>`. Built on-demand from note metadata, refreshed incrementally via timestamp queries. **No userData storage** to avoid sync conflicts and maintain per-note isolation philosophy.

**Performance Impact:**
* **Without index:** Load N notes → filter blocks → O(N) userData reads
* **With index:** nprobe lookup → load K notes where K = N × (nprobe / nlist) → O(K) userData reads
* **Speedup:** ~16x for typical parameters (nlist=64, nprobe=4)
* **Example:** 10,000 note corpus → load only ~625 notes instead of 10,000

**Design Philosophy:**

**Why in-memory only (no userData storage)?**
1. **Avoids centralization:** Storing index in anchor note violates userData's per-note isolation design
2. **Prevents sync conflicts:** Every note embedding would write to central anchor → conflicts
3. **Write amplification:** Update one note → read+modify+write 320KB index → expensive
4. **Graceful degradation:** Stale in-memory index has minimal impact (false positives caught by block filtering, false negatives resolved on next refresh)

**Data Structure (in-memory only):**

```typescript
class InvertedIndexCache {
  private index: Map<number, Set<string>>;  // centroidId → noteIds
  private lastUpdated: Date;                // timestamp of last refresh
  private nlist: number;                    // expected number of centroids
  
  // Methods: build(), refresh(), lookup(), updateNote()
}
```

**Build Strategy (full scan):**

```typescript
async build() {
  // Query ALL notes (no tag filter - most notes have embeddings)
  // Paginate in batches of 100 (Joplin API max limit per page)
  let page = 1;
  let hasMore = true;
  
  while (hasMore) {
    const response = await joplin.data.get(['notes'], {
      fields: ['id', 'updated_time'],
      limit: 100,
      page: page
    });
    
    // For each note: try to read metadata (cheap - returns null if no embeddings)
    for (const note of response.items) {
      const meta = await userDataStore.getMeta(note.id);
      if (!meta || !meta.models[activeModelId]) continue;
      
      // Has embeddings for active model - extract centroid IDs
      const shard = await userDataStore.getShard(note.id, 0);
      if (shard?.centroidIdsB64) {
        const centroidIds = decode_centroid_ids(shard.centroidIdsB64);
        for (const cid of unique(centroidIds)) {
          this.index.get(cid).add(note.id);
        }
      }
    }
    
    hasMore = response.has_more;
    page++;
  }
  
  this.lastUpdated = new Date();
}
```

**Note on tag usage:** Regular notes with embeddings are **not tagged** with `jarvis-database`. That tag is reserved only for infrastructure notes (System Catalog and per-model Anchors, typically <10 notes). Most notes (>95%) have embeddings but no special tag, so we must scan all notes and check for userData presence.

**Incremental Refresh Strategy (timestamp-based):**

```typescript
async refresh() {
  // Query notes updated since lastUpdated (catches synced + modified notes)
  // Paginate until updated_time <= lastUpdated
  let page = 1;
  let hasMore = true;
  
  while (hasMore) {
    const response = await joplin.data.get(['notes'], {
      fields: ['id', 'updated_time'],
      order_by: 'updated_time',
      order_dir: 'DESC',
      limit: 100,
      page: page
    });
    
    // Process only notes updated after lastUpdated
    for (const note of response.items) {
      if (note.updated_time <= this.lastUpdated) {
        // Reached notes older than last update - stop pagination
        hasMore = false;
        break;
      }
      
      // Remove old entries for this note (if any)
      this.removeNote(note.id);
      
      // Add new entries from current embeddings
      const meta = await userDataStore.getMeta(note.id);
      if (meta) this.updateNote(note.id, meta);
    }
    
    // Continue only if API has more AND we haven't reached old notes
    hasMore = hasMore && response.has_more;
    page++;
  }
  
  this.lastUpdated = new Date();
}
```

**Sync Detection:** The timestamp query (`order_by: updated_time, order_dir: DESC`) automatically catches:
- Newly synced notes (have recent `updated_time` on this device)
- Modified notes (local edits)
- Re-embedded notes (embeddings updated)

**Refresh Triggers (configurable based on performance):**

Options to tune based on performance logging:
1. **On every search:** ~50-100ms overhead for 10 updated notes (most sync-safe, slight latency cost)
2. **On note embedded:** Immediate in-memory update when local note embedded (free, but doesn't catch remote syncs)
3. **Periodic background:** Every 1-5 minutes (catches sync in batches, lower overhead)
4. **On manual "Update DB":** Full rebuild (rare, comprehensive)

**Recommendation:** Start with option 1 (on every search) and adjust based on measured latency. Timestamp query is very fast (~10-50ms for typical update rates).

**Search Flow Changes:**

**Before (current):**
```
1. Select top nprobe centroids
2. candidateIds = all notes in database
3. For each note in candidateIds:
     - Load metadata (N userData reads)
     - Load shard
     - Filter blocks by centroid ID
```

**After (with ephemeral index):**
```
1. [On startup: index built during database update - no search delay]
2. Refresh index (timestamp query, ~50-100ms typical)
3. Select top nprobe centroids → get centroidIds = [17, 42, 98, 133]
4. candidateIds = union(index[17], index[42], index[98], index[133]) → ~K notes
5. For each note in candidateIds (K << N):
     - Load metadata (K userData reads)
     - Load shard
     - Filter blocks by centroid ID (safety double-check)
```

**Build Timing:**

**Primary: Build during startup database update** - The app already runs a database update on startup. Build the inverted index as part of that process:
- Scan notes for embeddings updates (already happening)
- While scanning, accumulate centroid IDs per note
- After scan completes, index is ready
- Cost: amortized into existing startup process (no additional delay)
- First search is immediately fast

**Fallback: Lazy build on first search** - If startup update is skipped or index build fails:
- Build on first search (one-time cost, ~10-20s for 10K notes)
- Show progress indicator
- Subsequent searches are fast

**Edge Cases & Robustness:**

1. **Index not built yet:** Trigger build immediately during search (fallback path). Show progress indicator.

2. **Stale index (false positives):** Index says note has centroid, but it doesn't after update.
   - Load note → block filtering catches mismatch → skip blocks
   - Impact: minimal (a few extra notes loaded out of hundreds)

3. **Stale index (false negatives):** Index missing note that has centroid (just synced).
   - Note invisible until next `refresh()`
   - Duration: depends on refresh trigger (instant to 5 minutes)
   - Mitigated by refreshing on every search

4. **Deleted notes in stale index:** Try to load userData → 404 not found → skip gracefully (harmless).

5. **Multi-device scenarios:**
   - Each device builds own index independently
   - Timestamp queries ensure all devices converge to same state after sync
   - No conflicts (no shared artifact)

6. **Note has multiple blocks with different centroids:**
   - Note ID appears in multiple centroid sets
   - This is correct: when any of those centroids is probed, note is loaded
   - Block filtering during decode selects only matching blocks

7. **Empty centroids:** Some centroids may have no notes assigned. Store empty `Set()` for completeness (or omit from map).

**Memory & Performance Estimation:**

* **Memory:** ~1-2MB for 10,000 notes (Map + Sets in JavaScript)
* **Build cost:** ~10-20 seconds for 10K notes (one-time per session)
* **Refresh cost:** ~50-100ms for typical sync (10-50 updated notes)
* **Lookup cost:** O(nprobe) map lookups + set unions (~1ms)

**Fallback Strategy:**

```typescript
// If index is broken or refresh fails
if (!invertedIndex || !invertedIndex.isBuilt()) {
  // Fall back to current behavior: load all notes
  candidateIds = await getAllNotesWithEmbeddings();
  log.warn('Inverted index unavailable, using fallback (load all notes)');
}
```

**Implementation Priority:** **HIGH - CRITICAL FOR PERFORMANCE**

Without this, userData-based IVF search is significantly slower than SQLite for large corpora because every search loads all note metadata. With this ephemeral approach, userData achieves parity or better performance than SQLite while maintaining sync-friendly per-note isolation and avoiding sync conflicts entirely.

## 6) Algorithms

### 6.1 Block Segmentation

* Reuse the existing Markdown-aware block segmenter with configurable target token size (no additional overlap) so downstream "leading/trailing block" chaining keeps working.
* Allow per-model defaults to suggest sizes, but continue honoring user-configured limits.

### 6.2 Embedding & Quantization

* Compute F32 embedding per block → quantize to q8:

  * `int8 = clamp(round(float / scale), -127, 127)`
  * Store `scale` per row (Float32) or per block; store separately in shard.

### 6.3 Search

* **Query embedding:** compute F32 → quantize using the same policy (`scale_q`).
* **Candidate generation:**

  * Scope by notebook/tag/time window **or**
  * Use Joplin’s text search to shortlist top‑N note IDs, then vector‑rank within that set.
* **Auto‑switch rule:** if the candidate pool is **≤ 2k–5k rows**, skip IVF and do a **straight q8 scan** (often faster end‑to‑end). Above that, enable **IVF‑Flat**.
* **Ranking:** cosine similarity via q8 dot‑product + de‑quantization factor `(scale_row * scale_q)`.

### 6.4 IVF‑Flat (coarse quantizer)

**What it is:** A two‑stage index. Train **k‑means** centroids (**nlist**). Each vector is assigned to its nearest centroid (its **list**). At query time, compute distances to all centroids and **probe the top nprobe lists**; score only vectors in those lists.

**Why we use it:** It reduces work from **N** vectors to roughly `N × (nprobe / nlist)` with small memory overhead and a smooth latency/recall knob at query time.

**Storage:**

* Global centroids per `modelId` stored once in userData on the per-model anchor at `jarvis/v1/aux/centroids` (Float32/Float16, shape `[nlist, dim]`).
* For each embedded row, we store its **centroidId** alongside q8 bytes inside the shard payload.

**Search steps (shared desktop/mobile logic):**

1. Embed query (F32) → compute distances to centroids.
2. Pick top **nprobe** lists.
3. **[CRITICAL OPTIMIZATION]** Use inverted index to map selected centroid IDs → candidate note IDs. Only load userData for those notes (reduces from N notes to ~`N × (nprobe / nlist)` notes).
4. Stream shards for candidate notes; **decode only rows in the chosen lists**; compute q8 cosine; maintain a fixed‑size top‑K heap.

**Default heuristics:**

* Activate IVF when candidate pool > **5k** rows; otherwise brute‑force q8.
* Choose `nlist ≈ sqrt(Nmax)` to `2×sqrt(Nmax)` (e.g., Nmax≈20k → 256–512).
* Start with `nprobe ≈ 1–5%` of `nlist`; raise for higher recall.

**Platform knobs (same algorithm, different budgets):**

* **Mobile:** `candidateLimit = 300–600`, `nlist = 256–512`, `nprobe = 8–16`, `stopAfterMs = 60–120`.
* **Desktop:** `candidateLimit = 2k–10k`, `nlist = 512–1024`, `nprobe = 16–32`, larger time budgets (and optional local cache).

**Strengths:**

* Predictable latency control via `nprobe`.
* Modest extra memory (centroids only).
* Works well with q8 storage; inserts just update centroid ids.

**Weaknesses / gotchas:**

* Requires centroid training & occasional refresh if distributions shift.
* Boundary queries may need higher `nprobe` to recover recall.
* Non-uniform lists can hurt speed—mitigate by increasing `nlist`.
* Very small candidate pools: IVF overhead can exceed brute‑force → auto skip as above.

## 7) Write Path & Validation

### Write Path (on note change or forced rebuild)

1. On note save (or on‑demand):

   * Normalize + hash content → if `(modelId, contentHash)` unchanged and not forced, **skip**.
   * Run existing block segmentation; embed; quantize.
   * **Estimate shard size:** `rows * (dim + 4 + 2) * 1.33` (base64 overhead). If >500KB, **truncate to first N blocks that fit** and log warning.
   * **Pick a new `epoch`**: `(meta.models[modelId].current.epoch || 0) + 1`.
   * Write shard to `emb/<modelId>/live/0` with the new `epoch`.
   * Update `meta.models[modelId].current = { epoch, contentHash, rows, … }`.
   * Update `meta.activeModelId` if needed.
   * Add tag `jarvis-database` (idempotent).

### Validation Path (on every search)

**When loading embeddings for search:**

1. For each note with embeddings, validate:
   * `meta.activeModelId === currentModel.id`
   * `meta.models[activeModelId].embeddingVersion === currentModel.embedding_version`
   * `meta.models[activeModelId].modelVersion === currentModel.version`
   * `meta.models[activeModelId].settings` matches `currentSettings` (field-by-field comparison)

2. **For mismatched notes:** Include in search results anyway (use embeddings despite mismatch), but add to mismatch counter.

3. **After search completes:** If mismatches detected, show dialog **once per session**:
   * "Some notes have mismatched embeddings: [X notes: wrong model] [Y notes: different settings: embedTitle Yes→No, maxTokens 512→1024]. Rebuild affected notes now?"
   * Options: **[Rebuild Now]** (triggers `startUpdate(force: true, noteIds: mismatchedNoteIds)`) or **[Use Anyway]** (suppress dialog for session).

### Smart Rebuild Logic

* **Regular update (force=false):** Triggered by note save. Only rebuild notes where `contentHash` changed.
* **Forced update (force=true):** Triggered by settings change, manual "Update DB", model switch, or validation dialog. Rebuild notes where:
  * `contentHash` changed, OR
  * Settings mismatch with userData, OR
  * Model mismatch with userData
* **Granular per-note rebuilding:** Unlike SQLite (all-or-nothing), userData allows skipping already-up-to-date notes (e.g., synced from another device with new settings).

### Model Switching with Coverage Check

**When user changes `notes_model` setting:**

1. **Coverage check (fast sampling):**
   * Sample 100-200 random notes.
   * Check if `meta.models[newModelId]` exists in userData for each.
   * Calculate coverage: `notesWithModel / sampleSize`.

2. **If coverage < 10%:** Show dialog:
   * "Model '[modelId]' has ~[X]% coverage (estimated [count]/[total] notes). Populate embeddings now? This will use embedding API quota."
   * Options: **[Populate Now]** (set activeModelId, trigger rebuild), **[Switch Anyway]** (low coverage warning), **[Cancel]** (revert).

3. **If coverage ≥ 10%:** Seamless switch. Set `activeModelId`, log coverage estimate. On first search, validation checks settings/version compatibility and prompts if mismatches found.

**Concurrency/conflicts**

* Two devices may compute concurrently; the one that updates `meta.models[modelId].current` last becomes the winner. Readers always verify `shard.epoch === meta.models[activeModelId].current.epoch`.
* Because keys are fixed (`live/0`), there's no unbounded growth; at worst transient mixed epochs are ignored by readers.

---

## 8) Read Path per Platform

### Desktop

* Option A (simple): decode shards in memory on demand.
* Option B (fast): build a **local ANN cache** (HNSW/IVF) on disk from userData once, refresh incrementally; not synced.

### Mobile/Web

* No file storage; decode shards in memory for the session.
* Apply candidate shortlisting to keep latency acceptable.

Target budgets (mobile, mid‑range device):

* Shortlist 150–300 blocks, q8 cosine: ~10–40 ms.
* IVF‑Flat (256 lists, probe 12): ~5–20 ms typical.

---

## 8.5) Low‑memory mobile execution

**Budgets (defaults):** `maxDecodedRowsInMem = 4_000`, `candidateLimit = 400`, `topK = 20`.

**Techniques:**

* **Shard‑at‑a‑time decode:** decode one shard JSON → base64 → typed arrays → score → discard buffers (reuse a single work buffer).
* **Streaming shortlist:** generate candidates in waves (e.g., text prefilter, then IVF lists) so we never load all notes at once.
* **Reservoir top‑K:** maintain only a small heap of topK results; don’t store all scores.
* **LRU note cache (small):** keep recently decoded shards for interactive follow‑ups (e.g., size ≤ 2–4 shards).
* **Early exit:** stop scanning when `score < threshold` and remaining candidates’ upper bound is below current K‑th.

**Memory math example:**

* q8 row (512‑dim): ~512 B + minor scale data.
* 4,000 rows decoded simultaneously ≈ ~2–3 MB + JS overhead.
* One shard buffer (200 rows) ≈ ~120 KB; with reuse, peak stays dominated by the heap and a couple of shards.

## 9) Public API (inside Jarvis)

```ts
export type StoreKey = `jarvis/v1/emb/${string}/live/0`;

export interface EmbeddingSettings {
  embedTitle: boolean;
  embedPath: boolean;
  embedHeading: boolean;
  embedTags: boolean;
  includeCode: boolean;
  minLength: number;
  maxTokens: number;
}

export interface ModelMetadata {
  dim: number;
  metric: 'cosine'|'l2';              // L2 not yet supported, reserved for future
  modelVersion: string;
  embeddingVersion: number;
  settings: EmbeddingSettings;        // explicit settings instead of hash
  current: {
    epoch: number;
    contentHash: string;
    rows: number;
    blocking: { algo: string; avgTokens: number };
    updatedAt: string;
  };
}

export interface NoteEmbMeta {
  activeModelId: string;
  models: { [modelId: string]: ModelMetadata };
}

export interface ChunkRowMeta {
  title: string;
  headingLevel: number;
  headingPath?: string[];        // optional; omitted when it duplicates title
  bodyStart: number;
  bodyLength: number;
  lineNumber: number;
  // Removed: noteId, noteHash (inferred from parent), tags (embedded in vectors)
}

export interface EmbShard {
  epoch: number;
  format: 'q8';
  dim: number;
  rows: number;
  vectorsB64: string;              // base64 Int8Array
  scalesB64: string;               // base64 Float32Array (per row)
  centroidIdsB64?: string;         // base64 Uint16Array (optional)
  meta: ChunkRowMeta[];            // aligned to the encoded rows
}

export interface EmbStore {
  getMeta(noteId: string): Promise<NoteEmbMeta | null>;
  getShard(noteId: string, modelId: string): Promise<EmbShard>;  // always live/0
  put(noteId: string, modelId: string, meta: NoteEmbMeta, shard: EmbShard): Promise<void>; // desktop or capable devices
  gcModel(noteId: string, modelId: string): Promise<void>;
}
```

High‑level retrieval:

```ts
async function semanticSearch(query: string, scope: Scope): Promise<Result[]> {
  const candidateNoteIds = await shortlist(scope, query);       // tag/notebook/time or text search
  const rows = await loadRows(candidateNoteIds);                // lazy userDataGet, validate on load
  const q = embedQuery(query);                                  // F32
  const hits = rankQ8Cosine(q, rows, { ivf: maybeCentroids });
  
  // Validation: check mismatches, show dialog if needed
  if (mismatches.length > 0 && !sessionDialogShown) {
    showMismatchDialog(mismatches);
  }
  
  return topK(hits, scope.k);
}
```

---

## 10) Settings & Controls

* **Model selection** (chat vs index) and `modelId` mapping with multi-model support.
* **Indexing policy:** on-save / manual / schedule; per-notebook opt-out via `jarvis-exclude` tag (backward compatible with `exclude.from.jarvis`).
* **Unified search engine knobs:**
  * **Auto IVF** (on/off): enable IVF when candidate pool > threshold.
  * **Small-scope threshold** (default 2k–5k) for brute-force q8.
  * **IVF params:** `nlist` (advanced), `nprobe` (slider), per-platform presets.
  * **Candidate cap** and **time budget** (`stopAfterMs`).
  * **Mobile memory budget** (e.g., `maxDecodedRowsInMem`).
* **Privacy:** option to **never** write embeddings for specific notebooks/tags (enforced via `jarvis-exclude`).
* **Maintenance:** 
  * "Rebuild embeddings for this notebook" (force rebuild)
  * "Update DB" with smart rebuild (only outdated notes)
* **Model cleanup:** Management dialog listing stored embedding models with:
  * Model ID, version, note count, approximate storage, last used timestamp, active status
  * Delete action: removes `meta.models[modelId]` entries and `jarvis/v1/emb/<modelId>/live/0` shards
  * Includes associated anchor notes and centroids in deletion
* **Settings change handling:** When embedding settings change, prompt user to rebuild with human-readable diff showing exactly what changed (e.g., "embedTitle Yes→No, maxTokens 512→1024").
* **Model switching:** Coverage check with sampling (100-200 notes) to estimate percentage with new model. Low coverage (<10%) triggers prompt; high coverage seamlessly switches.

---

## 11) Migration Plan

**Automatic per-model migration from SQLite to userData:**

* **Migration strategy:** Each model's SQLite is migrated independently when that model is activated/loaded (not all at startup).
* **Trigger point:** When user activates a model (via settings), check if migration needed for that specific model.
* **Priority resolution:** If userData exists for a note/model, skip SQLite migration (userData takes precedence).

**Per-model migration flow:**

1. Check plugin settings: `migrationCompletedForModel[modelId]` boolean flag.
2. If `true` → skip migration for this model (already done).
3. If `false` or undefined → check if SQLite file exists at `{pluginDir}/{modelId}.sqlite`.
4. If SQLite exists → trigger automatic background migration for this model (no user prompt).
5. Show progress in panel: "Migrating model '{modelId}' embeddings to new format... 45/120 notes"

**Migration logic per note for this model:**

1. Check if `meta.models[modelId]` already exists in userData → skip (already migrated or synced from another device).
2. If not: read from SQLite for this model, validate hash matches current note content.
3. If hash matches: write to userData with current format (add to `meta.models[modelId]`).
4. If hash doesn't match: skip (note changed since SQLite embedding, will be rebuilt on next update).

**Progress tracking per model:**

* Store in plugin settings: `migrationCompletedForModel: { [modelId]: boolean }`
* Example: `{ "use-512": true, "openai-1536": false }` means first model migrated, second not yet.
* Set to `true` only after full migration completes for that model.
* If migration interrupted (app closed), restart when model next activated is cheap: userData-first check skips already-migrated notes (just metadata reads).

**Multiple SQLite databases:** Since each model has its own SQLite file, migrations are independent. User can have model A fully migrated while model B still uses SQLite.

**Post-migration:**

* Keep SQLite file intact (for rollback) until soak period passes.
* After soak period, optionally archive or delete legacy `.sqlite` files via maintenance dialog.
* **Post-migration validation:** Spot-check 10 random notes per model: compare search results between SQLite and userData (before SQLite deletion).
* Tag migrated notes with `jarvis-database` for discovery.
* Mobile/web automatically consume userData without any migration work.

---

## 12) Security & Privacy

* Embeddings live in **note‑scoped userData**, invisible in editors.
* They inherit Joplin's sync/e2ee semantics for note‑associated data.
* Per‑notebook **denylist** via `jarvis-exclude` tag (backward compatible with `exclude.from.jarvis`) ensures sensitive notebooks are never embedded.

---

## 12.5) Graceful Degradation & Error Handling

**Format-agnostic metadata handling:**

* When reading `jarvis/v1/meta`, wrap in try-catch and handle parse failures gracefully.
* If metadata fails to parse or has unexpected structure (missing required fields, wrong types), log warning and treat as "no embeddings".
* On next update for that note, fresh metadata will be written with current format (effectively migrating forward).
* No need to support specific old formats—just fail gracefully and rebuild on next opportunity.

**Handling corrupted or missing data:**

* **Missing embeddings:** Skip note silently, continue search with other notes. Log count of skipped notes.
* **Corrupted metadata (malformed JSON):** Skip note, log warning with note ID. User can rebuild via "Update DB".
* **Corrupted shard data (invalid base64):** Skip note, log warning. User can rebuild via "Update DB".
* **Missing centroid data:** Fallback to brute-force search (no IVF), log warning. Slower but functional.
* **userData read failure:** Skip note for this search cycle, retry on next search. Log transient error.
* **Shard exceeds size limit:** Show user dialog with note title and size, offer to skip or split note manually.

**Transient errors during sync:**

* If note metadata seems inconsistent during search (e.g., sync in progress), log clearly: "Note ${noteId}: inconsistent metadata (may be rebuilding on another device or sync in progress), skipping"
* Don't fail hard—graceful degradation keeps search functional even with partial data.

---

## 13) Telemetry (opt‑in)

* Count of notes embedded, average shards/rows, average query latency, IVF probe count.
* No content, no vectors, no note titles in telemetry.

---

## 14) Testing Strategy

* **Unit:** block segmenter, quantizer, q8 cosine, IVF routing, base64 encode/decode.
* **Integration:** 
  * write→read round‑trip via userData on all platforms
  * multi‑device sync convergence (same content → identical keys/values)
  * multi-model coexistence (different models on same note)
  * validation logic (detect mismatches, show correct dialogs)
* **Performance:**
  * **Validation overhead benchmark:** Real corpus with 1000 notes. Measure total validation time during search (metadata reads + field comparisons). Target: <50ms total for 1000 notes (avg <0.05ms per note).
  * **Mobile memory:** enforce `maxDecodedRowsInMem`; shard-at-a-time decode; verify peak RSS on target devices.
  * **Latency:** measure IVF vs brute-force crossover (threshold 2k–5k) on representative corpora; tune `nprobe`.
  * **IVF vs brute-force latency crossover:** Varying corpus sizes (100, 500, 1000, 5000 notes), different `nlist`/`nprobe` configurations. Identify crossover point (~500+ notes expected).
  * **nprobe tuning sweeps:** Test different `nprobe` values (1, 2, 4, 8, 16) on 1000-note corpus. Measure recall@10 vs latency trade-off.
  * **500KB threshold validation:** Collect metrics on notes approaching cap during beta (>400KB). Track note size distribution. Validate threshold affects <1% of notes.
* **Quality:** 
  * recall@K vs brute-force F32/q8 baselines
  * run `nprobe` sweeps to hit ≥0.95 recall@10
  * search relevance regression tests (userData q8 vs legacy SQLite F32)
* **Upgrade:** 
  * change `modelId`/`dim`; ensure old keys survive until GC
  * multi-model preservation: create embeddings with model A, switch to B, verify both intact
* **Real-world scenarios:**
  * Two devices with different models (coexistence)
  * Two devices with same model/settings (happy path sync)
  * Two devices with same model, different settings (validation dialog)
  * Model switch with low coverage (coverage dialog)
  * Concurrent rebuild race conditions
  * Large note exceeding 500KB cap
  * Migration from SQLite (single device, multi-device coordination)

---

## 15) Rollout Plan

* **Phase 0:** behind a feature flag (`notes_db_in_user_data`).
* **Phase 1 (desktop write + mobile read):** enable writes on desktop; mobile reads/queries.
* **Phase 2:** optional desktop local ANN cache; add IVF centroids for faster mobile.
* **Phase 3:** deprecate legacy stores; enable GC.

---

## 16) Risks & Mitigations

* **Large userData blobs → sync overhead:** mitigate with q8, single-shard constraint (500KB cap with automatic truncation), and shortlist search.
* **Inconsistent embeddings across devices (different settings/models):** mitigate with validation on every search and user prompts to rebuild mismatched notes.
* **Multi-model coordination:** each model has independent metadata and shards. `activeModelId` selector ensures queries use correct model.
* **User disables tags or edits note titles:** discovery uses tags (`jarvis-database`) *or* cached registry; fall back to scanning on demand.
* **Model upgrade churn:** versioned keys per model; staged migration with per-model tracking; smart rebuilds skip already-up-to-date notes.
* **Very large notes (>500KB):** automatically truncated during embedding with warnings; user can split note manually if needed.
* **Sync conflicts:** last-writer-wins via `epoch` and `updatedAt` timestamp; readers validate epoch consistency.

---

## 17) Open Questions

1. Maximum practical size per `userData` value across sync providers? (**Resolved:** Target 500KB per note with automatic truncation.)
2. E2EE guarantees for note‑scoped userData — confirm equivalence with note content encryption.
3. API rate/throughput limits for `userDataGet/Set` on mobile/web.
4. Should we standardize IVF centroids per `modelId` (global) or train per‑library? (**Decision:** Global per-model centroids with canonical `nlist=2048`.)
5. Validation overhead acceptable? (**To benchmark:** <50ms for 1000 notes target during beta.)
6. Should validation run on every search or less frequently? (**Decision:** Every search initially, reconsider if overhead >50ms.)
7. Is 500KB cap appropriate? (**To validate:** Collect metrics during beta; should affect <1% of notes.)
8. Should 500KB overflow trigger dialog or automatic truncation? (**Decision:** Automatic truncation with warnings for MVP; can add dialog later if needed.)

---

## 18) Appendix

### A. Normalization for `contentHash`

* Reuse existing normalization pipeline: convert HTML to Markdown when needed and normalize newlines to `\n`.
* Avoid additional stripping or casing changes to preserve compatibility with existing hashes unless a dedicated migration is planned.

### B. Example sizes (rule‑of‑thumb)

* 512‑dim F32 = 2 KB per block; q8 ≈ 512 B + scale (4 B) → ~516 B/row before base64.
* 1536‑dim q8 ≈ 1536 B + scale (4 B) → ~1540 B/row before base64 → ~2053 B/row after base64 (×1.33).
* **500KB single-shard capacity:**
  * 512-dim: ~500 blocks per note
  * 1536-dim: ~220 blocks per note
* **Size estimation formula:** `rows × (dim + 4 + 2) × 1.33` (includes q8 bytes, scale, centroid ID, base64 overhead).
* Shard size checked **before** calling embedding API to avoid wasting quota on blocks that won't fit.

### C. Minimal Inspect Tools (dev menu)

* "Show embeddings meta for this note" (displays active model and all available models)
* "Show all models for this note" (list all `meta.models` entries with stats)
* "Re‑embed now" (force rebuild with current active model)
* "Delete model embeddings for this note" (remove specific model from `meta.models`)
* "Simulate mobile query path" (test mobile presets on desktop)
* "Validate embeddings" (run validation checks manually, show mismatches)
* "Show migration status" (display `migrationCompletedForModel` flags)
