# Database Update and Caching Flow

This document describes when database updates are triggered and what gets updated/cached as a result.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JARVIS DATABASE SYSTEM                      │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐   ┌──────────────┐    │
│  │  Note UserData   │    │  Catalog Note    │   │  RAM Cache   │    │
│  │  (embeddings)    │◄───│  (metadata)      │◄──│  (Q8 search) │    │
│  └──────────────────┘    └──────────────────┘   └──────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Trigger Events and Consequences

### 1. Plugin Startup

```mermaid
graph TD
    A[Plugin Loads] --> B[Initialize Models]
    B --> C{DB Exists?}
    C -->|No| D[Full Sweep]
    C -->|Yes| E[Incremental Sweep]
    D --> F[Schedule Timer]
    E --> F
    F --> G[Ready]

    D -->|Updates| H[UserData Embeddings]
    D -->|Updates| I[Catalog Metadata]
    D -->|Updates| J[RAM Cache<br/>Incrementally]

    E -->|Updates| K[UserData Embeddings<br/>Changed Notes Only]
    E -->|No Update| L[Catalog Metadata<br/>Unchanged]
    E -->|Updates| M[RAM Cache<br/>Incrementally]
```

**What happens:**
- **Initial sweep** runs in background
  - First time: Full sweep (scans all notes)
  - Subsequent: Incremental sweep (only changed notes)
- **Timer scheduled** for periodic sweeps
- **Cache NOT built** yet (waits for first search)

**What gets updated:**
- ✅ UserData embeddings (new/changed notes only if incremental)
- ✅ Catalog metadata (only if full sweep)
- ❌ RAM cache (not built until first search)

---

### 2. Periodic Timer (Every N Minutes)

```mermaid
graph TD
    A[Timer Fires] --> B{Update in Progress?}
    B -->|Yes| C[Skip]
    B -->|No| D{Last Full Sweep > 12h?}
    D -->|Yes| E[FULL SWEEP]
    D -->|No| F[INCREMENTAL SWEEP]

    E --> G[Scan ALL Notes]
    G --> H[Update Changed Notes]
    H --> I[Count All Notes]
    I --> J[Update Catalog Metadata]
    J --> K[Incrementally Update Cache]

    F --> L[Query Recent Notes<br/>by Timestamp]
    L --> M[Stop at Old Notes]
    M --> N[Update Changed Notes]
    N --> O{Settings Mismatch?}
    O -->|Yes| P[Show Dialog to User]
    P --> Q{User Choice}
    Q -->|Rebuild| R[Trigger Full Sweep]
    Q -->|Skip| S[Continue]
    O -->|No| S
    S --> T[Preserve Metadata]
    T --> U[Incrementally Update Cache]
```

**Decision Logic:**
- Full sweep needed if: never done before OR more than 12 hours since last full sweep
- Otherwise: incremental sweep (only recently changed notes)

**What gets updated:**

**Full Sweep (every 12 hours):**
- ✅ UserData embeddings (all changed notes)
- ✅ Catalog metadata (rowCount, noteCount, dim)
- ✅ lastFullSweepTime timestamp
- ✅ RAM cache (incrementally updated for all processed notes)
- ✅ Exclusion cleanup (removes embeddings for newly excluded notes)

**Incremental Sweep (rest of the time):**
- ✅ UserData embeddings (recently changed notes only)
- ✅ RAM cache (incrementally updated for all processed notes, including unchanged ones)
- ✅ Settings validation (checks embeddings match current settings)
- ✅ lastSweepTime timestamp
- ⚠️ **User dialog** if settings mismatches detected (e.g., synced from another device)
  - User can choose to rebuild mismatched notes immediately
  - Or continue using mismatched embeddings
- ❌ Catalog metadata (unchanged unless user triggers rebuild)

**Cache Strategy:**
- Incremental sweeps keep cache fresh by updating it for ALL processed notes (even unchanged ones)
- This catches newly synced notes without expensive full cache rebuilds
- Fast and unnoticeable on mobile (vs 20-30 sec for full cache rebuild)

---

### 3. User Edits Note

```mermaid
graph TD
    A[Note Saved] --> B[Add to pending_note_ids]
    B --> C[Debounce 5s]
    C --> D{Note Still Pending?}
    D -->|Yes| E[Update Specific Note]
    D -->|No| F[Skip Already Processed]

    E --> G[Re-embed Note]
    G --> H[Write to UserData]
    H --> I{Cache Built?}
    I -->|Yes| J[Incrementally Update Cache<br/>Replace note's blocks]
    I -->|No| K[Skip Cache Update]

    H --> L[No Metadata Update]
    H --> M[No lastSweepTime Update]
```

**What happens:**
- Note ID added to `pending_note_ids` set
- Debounced update triggered (5 second delay)
- Only the specific note is re-embedded
- Cache is **incrementally updated** (not rebuilt)

**What gets updated:**
- ✅ UserData embeddings (this note only)
- ✅ RAM cache (incrementally - note's blocks replaced)
- ❌ Catalog metadata (unchanged - drift corrected by daily full sweep)
- ❌ lastSweepTime (unchanged)

---

### 4. User Searches for Related Notes

```mermaid
graph TD
    A[User Searches] --> B{Cache Exists?}
    B -->|No| C[Build Cache]
    B -->|Yes| D{Cache Valid?<br/>Dim Matches?}
    D -->|No| E[Rebuild Cache]
    D -->|Yes| F[Use Existing Cache]

    C --> G[Fetch All Notes<br/>Paginated]
    G --> H[Tag Reverse-Lookup<br/>4 API calls]
    H --> I[Filter Candidates<br/>In-Memory]
    I --> J[Try Load Shards<br/>Graceful Failures]
    J --> K[Store Q8 in RAM]
    K --> L[Set Model Stats<br/>Actual Count]
    L --> M[Search in RAM]

    E --> G
    F --> M

    M --> N[Return Results<br/>10-50ms]
```

**Cache Build Process (Two Phases):**

Only builds cache when needed (first search or dimension change).

**Phase 1: Fetch and Filter (silent - fast)**
1. Fetch exclusion tags once via reverse-lookup
2. Paginate through all notes (100 per page)
3. For each note: count total, check exclusions, collect candidate IDs
4. Track: totalNotes, excludedCount, candidateIds

**Phase 2: Load Embeddings (with progress)**
1. Try loading shards for all candidate IDs
2. `getShard` returns null if no embeddings → skip gracefully
3. Update progress bar every 10 notes loaded
4. Progress formula: `(excludedCount + loaded) / totalNotes`
5. Example: 63 excluded, 999 with embeddings, 1062 total
   - Shows: 73/1062, 163/1062, ..., 1062/1062

**Subsequent searches:**
- Skip both phases entirely
- Pure RAM search using cached Q8 vectors (10-50ms)

**Performance (example: 5000 notes, 3000 with embeddings):**

| Step | API Calls | Description |
|------|-----------|-------------|
| Fetch notes (paginated) | 50 | 100 notes per page |
| Tag reverse-lookup | 4 | Query notes by exclusion tags |
| Filter candidates | 0 | In-memory using response data |
| Try load metadata | 4,500 | One per candidate (inside getShard) |
| Load shards | 3,000 | Only for notes with embeddings |
| **Total** | **7,554** | **~45 seconds** |

**Key characteristics:**
- ✅ **Tag reverse-lookup**: Single query per exclusion tag (not per note)
- ✅ **Graceful failure handling**: getShard returns null if no embeddings exist
- ✅ **Single-pass architecture**: Metadata fetched once (inside getShard)
- ✅ **Two-phase progress**: Phase 1 silent (fast fetch/filter), Phase 2 shows progress (embedding load)
- ✅ **Total notes denominator**: Progress bar always shows X/1062 (matches full sweep UX)

**What happens:**
- **First search:** Cache built (2-5 seconds on desktop, 5-15 seconds on mobile)
- **Subsequent searches:** Pure RAM search (10-50ms)
- **Progress indicator:** Phase 1 silent (fast), Phase 2 shows X/totalNotes progress every 10 embeddings loaded

**What gets updated:**
- ✅ RAM cache (all Q8 vectors for notes with embeddings)
- ✅ Model stats in memory (actual rowCount, noteCount, dim)
- ❌ UserData embeddings (unchanged)
- ❌ Catalog metadata (unchanged)

---

### 5. Manual "Update DB" Command

```mermaid
graph TD
    A[User Clicks Update DB] --> B{force = true}
    B --> C[FULL SWEEP]
    C --> D[Scan ALL Notes]
    D --> E[Skip if Content + Settings Match]
    E --> F[Rebuild if Mismatch]
    F --> G[Update Catalog Metadata]
    G --> H[Invalidate Cache]
```

**What happens:**
- Forces **full sweep** with `force=true`
- Checks content hash + settings + model version
- Rebuilds notes where any mismatch detected
- Always updates catalog metadata

**What gets updated:**
- ✅ UserData embeddings (all mismatched notes)
- ✅ Catalog metadata (always updated)
- ✅ lastFullSweepTime (reset)
- ⚠️ RAM cache (invalidated, rebuilds on next search)

---

## Settings Mismatch Detection

The system automatically detects when embeddings were created with different settings than the current configuration. This commonly happens when syncing between devices with different settings.

```mermaid
graph TD
    A[Sweep Processes Note] --> B{Content Changed?}
    B -->|Yes| C[Rebuild Embeddings]
    B -->|No| D{Settings Match?}
    D -->|Yes| E[Skip Note]
    D -->|No| F[Collect Mismatch Info]

    C --> G[Sweep Continues]
    E --> G
    F --> G

    G --> H[After Sweep Completes]
    H --> I{Mismatches Found?}
    I -->|No| J[Done]
    I -->|Yes| K[Show Dialog to User]

    K --> L{User Decision}
    L -->|Rebuild| M[Full Sweep force=true<br/>Rebuilds All Mismatches]
    L -->|Use Anyway| N[Continue with Mismatches]
```

**Detection Timing:**

| Scenario | Detection Window | Action |
|----------|-----------------|---------|
| **Recently synced notes** | Next incremental sweep (≤30 min) | User prompted immediately |
| **Old synced notes** | Daily full sweep (≤12h) | User prompted |
| **Settings changed locally** | Immediate | Force rebuild triggered |

**What gets checked:**
- `embedTitle` (include note title in embeddings)
- `embedPath` (include folder path in embeddings)
- `embedHeading` (include heading in embeddings)
- `embedTags` (include tags in embeddings)
- `includeCode` (include code blocks)
- `maxTokens` (maximum tokens per block)

**Mismatch dialog:**
- Shows count of mismatched notes and which settings differ
- User can choose to rebuild immediately or continue with mismatches
- Example: "Found 15 note(s) with different embedding settings (likely synced from another device)"

**Important notes:**
- Only checks notes that are processed during the sweep (recently changed or all notes in full sweep)
- Old, unchanged notes are only validated during daily full sweeps (every 12 hours)
- Mismatches don't affect correctness, just embedding quality/consistency
- User can choose to rebuild immediately or defer to next full sweep

---

## Cache Invalidation Rules

The RAM cache is invalidated (cleared) in these cases:

```mermaid
graph TD
    A{Cache Invalidation?} --> B[Excluded Folders Changed]
    A --> C[Model Changed]
    A --> D[Dimension Changed]
    A --> E[Delete All Models]

    B --> F[Cache.invalidate]
    C --> F
    D --> F
    E --> F
```

**Invalidation triggers (complete rebuild on next search):**
1. ❌ **Excluded folders changed** (settings change)
2. ❌ **Model switched** (user selects different model)
3. ❌ **Dimension mismatch** (model updated with different dim)
4. ❌ **Delete all models** (user action)

**NOT invalidated by (incremental updates instead):**
- ✅ Full sweeps (cache updated incrementally for all processed notes)
- ✅ Incremental sweeps (cache updated incrementally for recently changed notes)
- ✅ Individual note updates (cache updated incrementally)
- ✅ Note edits (cache updated incrementally)
- ✅ Note deletions (blocks removed incrementally)
- ✅ Sync operations (cache kept fresh via incremental sweeps)

**Note about exclusion tags:**
- Adding/removing `jarvis-exclude` tag does NOT invalidate cache
- **If cache rebuilt:** New exclusions picked up immediately during Phase 1 filtering
- **If note processed by sweep:** Embeddings removed from userData and cache incrementally
- **If cache not rebuilt AND note not processed by sweep:** Excluded note may remain in search results until:
  - Next cache rebuild (triggered by dimension change, excluded folder change, model switch), OR
  - Next full sweep (every 12 hours) processes the note

---

## Data Flow Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          TRIGGER EVENTS                            │
├────────────┬───────────────┬──────────────┬────────────────────────┤
│  Startup   │  Timer (12h)  │  Note Edit   │  Search                │
└────┬───────┴───────┬───────┴──────┬───────┴────────┬───────────-───┘
     │               │              │                │
     │               ▼              ▼                ▼
     │         ┌──────────┐   ┌──────────┐    ┌──────────┐
     │         │   FULL   │   │ SPECIFIC │    │  CACHE   │
     └────────►│  SWEEP   │   │   NOTE   │    │  BUILD   │
               │ (daily)  │   │  UPDATE  │    │ (first)  │
               └────┬─────┘   └────┬─────┘    └────┬─────┘
                    │              │               │
     ┌──────────────┼──────────────┼───────────────┤
     │              │              │               │
     ▼              ▼              ▼               ▼
┌─────────-┐   ┌─────────┐   ┌─────────┐   ┌──────────────┐
│ UserData │   │ Catalog │   │Incr.    │   │ Model Stats  │
│Embeddings◄──-┤Metadata │   │Cache    │   │  (in-mem)    │
│(per note)│   │(global) │   │Update   │   │              │
└─────────-┘   └─────────┘   └─────────┘   └──────────────┘
     │              │              │               │
     └──────────────┴──────────────┴───────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │  RAM CACHE   │
                  │ (Q8 vectors) │
                  │   10-50ms    │
                  └──────────────┘
```

---

## Update Frequency Summary

| Event | Frequency | Full Sweep? | Metadata? | Cache? | Settings Check? |
|-------|-----------|-------------|-----------|--------|-----------------|
| **Startup** | Once | Only if first time | Only if full | Not built | Only if full |
| **Periodic (Timer)** | Every N min | Once per 12h | Once per 12h | Incrementally updated | Yes (all processed notes) |
| **Note Edit** | Per save | No | No | Incrementally updated | No |
| **Search** | On demand | No | No | Built if needed | No |
| **Manual Update** | User action | Yes | Yes | Incrementally updated | Yes (all notes) |

---

## Key Decision Points

**When to do full sweep:**
- Never done before (lastSweepTime === 0), OR
- More than 12 hours since last full sweep

**When to build cache:**
- Cache not built yet, OR
- Dimension mismatch (model changed)

**When to update catalog metadata:**
- Full sweeps only (not incremental sweeps)
