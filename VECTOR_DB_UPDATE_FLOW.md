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
    D -->|Invalidates| J[RAM Cache]

    E -->|Updates| K[UserData Embeddings<br/>Changed Notes Only]
    E -->|No Update| L[Catalog Metadata<br/>Unchanged]
    E -->|No Invalidation| M[RAM Cache<br/>Preserved]
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
    B -->|No| D{Last Full Sweep > 24h?}
    D -->|Yes| E[FULL SWEEP]
    D -->|No| F[INCREMENTAL SWEEP]

    E --> G[Scan ALL Notes]
    G --> H[Update Changed Notes]
    H --> I[Count All Notes]
    I --> J[Update Catalog Metadata]
    J --> K[Invalidate Cache]

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
    T --> U[Preserve Cache]
```

**Decision Logic:**
```typescript
const lastFullSweepTime = await get_model_last_full_sweep_time(modelId);
const now = Date.now();
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

const needsFullSweep = lastSweepTime === 0 || (now - lastFullSweepTime) > ONE_DAY_MS;
```

**What gets updated:**

**Full Sweep (once per day):**
- ✅ UserData embeddings (all changed notes)
- ✅ Catalog metadata (rowCount, noteCount, dim)
- ✅ lastFullSweepTime timestamp
- ⚠️ RAM cache (invalidated, rebuilds on next search)

**Incremental Sweep (rest of the time):**
- ✅ UserData embeddings (recently changed notes only)
- ✅ Settings validation (checks embeddings match current settings)
- ✅ lastSweepTime timestamp
- ⚠️ **User dialog** if settings mismatches detected (e.g., synced from another device)
  - User can choose to rebuild mismatched notes immediately
  - Or continue using mismatched embeddings
- ❌ Catalog metadata (unchanged unless user triggers rebuild)
- ❌ RAM cache (preserved unless user triggers rebuild)

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

    C --> G[Fetch Note IDs with Embeddings]
    G --> H[Load All Q8 Vectors]
    H --> I[Store in RAM]
    I --> J[Set Model Stats]
    J --> K[Search in RAM]

    E --> G
    F --> K

    K --> L[Return Results<br/>10-50ms]
```

**Cache Build Process:**
```typescript
// Only fetches note IDs when cache needs building
if (!cache.isBuilt() || cache.getDim() !== queryDim) {
  const result = await get_all_note_ids_with_embeddings(modelId, ...);
  await cache.ensureBuilt(...);
}

// Subsequent searches skip this entirely
const results = cache.search(queryQ8, k, minScore);
```

**What happens:**
- **First search:** Cache built (2-5 seconds)
- **Subsequent searches:** Pure RAM search (10-50ms)

**What gets updated:**
- ✅ RAM cache (all Q8 vectors loaded)
- ✅ Model stats in memory (rowCount, noteCount, dim)
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
| **Old synced notes** | Daily full sweep (≤24h) | User prompted |
| **Settings changed locally** | Immediate | Force rebuild triggered |

**What gets checked:**
- `embedTitle` (include note title in embeddings)
- `embedPath` (include folder path in embeddings)
- `embedHeading` (include heading in embeddings)
- `embedTags` (include tags in embeddings)
- `includeCode` (include code blocks)
- `maxTokens` (maximum tokens per block)

**Mismatch dialog example:**
```
Found 15 note(s) with different embedding settings (likely synced from another device).

Settings: embedTitle No→Yes, maxTokens 512→1024

Rebuild these notes with current settings?
[OK] [Cancel]
```

**Important notes:**
- Only checks notes that are processed during the sweep (recently changed or all notes in full sweep)
- Old, unchanged notes are only validated during daily full sweeps
- Mismatches don't affect correctness, just embedding quality/consistency
- User can choose to rebuild immediately or defer to next full sweep

**Code references:**
- Detection: embeddings.ts:613-629
- Dialog: notes.ts:396-422

---

## Cache Invalidation Rules

The RAM cache is invalidated (cleared) in these cases:

```mermaid
graph TD
    A{Cache Invalidation?} --> B[Full Sweep Starts]
    A --> C[Dimension Changed]
    A --> D[Note Deleted]

    B --> E[Cache.invalidate]
    C --> E
    D --> F{Cache Built?}
    F -->|Yes| G[Incremental Update<br/>Remove note's blocks]
    F -->|No| E
```

**Invalidation triggers:**
1. ❌ **Full sweep starts** → Complete rebuild on next search
2. ❌ **Model changed** (dimension mismatch) → Complete rebuild
3. ⚠️ **Note deleted** → Incremental update (remove blocks)

**NOT invalidated by:**
- ✅ Incremental sweeps
- ✅ Individual note updates
- ✅ Note edits (incrementally updated instead)

---

## Data Flow Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          TRIGGER EVENTS                            │
├────────────┬───────────────┬──────────────┬────────────────────────┤
│  Startup   │  Timer (24h)  │  Note Edit   │  Search                │
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
| **Periodic (Timer)** | Every N min | Once per 24h | Once per 24h | Invalidated daily | Yes (all processed notes) |
| **Note Edit** | Per save | No | No | Incrementally updated | No |
| **Search** | On demand | No | No | Built if needed | No |
| **Manual Update** | User action | Yes | Yes | Invalidated | Yes (all notes) |

---

## Code References

### Key Functions

- **Full Sweep:** `update_note_db(..., incrementalSweep=false)` → notes.ts:232
- **Incremental Sweep:** `update_note_db(..., incrementalSweep=true)` → notes.ts:172
- **Cache Build:** `cache.ensureBuilt(...)` → embeddingCache.ts:153
- **Cache Search:** `cache.search(queryQ8, k, minScore)` → embeddingCache.ts:278
- **Metadata Update:** `write_model_metadata(...)` → notes.ts:340
- **Timer Logic:** `schedule_full_sweep_timer(...)` → index.ts:267

### Decision Points

```typescript
// When to do full sweep (index.ts:289)
const needsFullSweep = lastSweepTime === 0 || (now - lastFullSweepTime) > ONE_DAY_MS;

// When to fetch note IDs (embeddings.ts:1360)
if (!cache.isBuilt() || cache.getDim() !== queryDim) {
  const result = await get_all_note_ids_with_embeddings(...);
}

// When to update metadata (notes.ts:311)
if (isFullSweep && !incrementalSweep) {
  await write_model_metadata(...);
}
```
