# Jarvis Vector Database: Technical Specification

**Version**: 3.1
**Status**: Experimental (behind `notes_db_in_user_data` flag)
**Last Updated**: 2025-12-04

> **Note**: Version 3.1 removed redundant metadata fields and flattened the storage structure (YAGNI cleanup). Version 3.0 removed IVF (Inverted File Index) in favor of simpler in-memory brute-force search. These changes significantly simplify the architecture while maintaining excellent performance for typical note collections.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
   - Design Philosophy
   - High-Level Architecture
   - Key Components
2. [Storage Design](#2-storage-design)
   - Storage Hierarchy
   - Storage Keys
   - Single-Shard Constraint
3. [Data Schemas](#3-data-schemas)
   - 3.1 NoteEmbMeta (Per-Note Metadata)
   - 3.2 EmbShard (Per-Note Embeddings)
   - 3.3 CatalogMetadata (Per-Model Metadata)
4. [Algorithms](#4-algorithms)
   - 4.1 Block Segmentation
   - 4.2 Quantization (Q8)
   - 4.3 In-Memory Cache
5. [Search Pipeline](#5-search-pipeline)
   - Full Search Flow
   - Search Tuning Parameters
6. [Performance Characteristics](#6-performance-characteristics)
   - Time Complexity
   - Space Complexity
7. [Platform-Specific Optimizations](#7-platform-specific-optimizations)
   - Desktop
   - Mobile
8. [Implementation Details](#8-implementation-details)
   - Write Path
   - Read Path
   - Cache Management
9. [Future Enhancements](#9-future-enhancements)
10. [Implementation References](#10-implementation-references)

---

## 1. Architecture Overview

### Design Philosophy

The Jarvis vector database is designed as a **sync-first, per-note embedded index** that stores semantic embeddings directly in Joplin note metadata (userData) instead of in a centralized SQLite database. This approach enables:

- **Cross-device synchronization**: Embeddings sync automatically via Joplin's native sync
- **Per-device model selection**: Each device can use a different active model (e.g., desktop uses OpenAI, mobile uses local model)
- **Mobile support**: No filesystem access required; runs on mobile, web, and desktop
- **Multi-model coexistence**: Each note can have embeddings from multiple models simultaneously
- **Automatic backup**: Embeddings are backed up with notes (stored in userData, not separate files)
- **Incremental updates**: Per-note isolation allows independent updates without full rebuilds
- **Conflict-free operation**: Content-addressed epochs prevent sync conflicts

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Jarvis Plugin                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌────────────────────────┐       │
│  │  Embedding   │─────▶│  UserData EmbStore     │       │
│  │  Indexer     │      │  (per-note storage)    │       │
│  └──────────────┘      └────────────────────────┘       │
│         │                         │                     │
│         │                         ▼                     │
│         │              ┌────────────────────────┐       │
│         │              │  In-Memory Cache       │       │
│         │              │  (Q8 vectors + meta)   │       │
│         │              │  100MB mobile/200MB    │       │
│         │              └────────────────────────┘       │
│         │                         │                     │
│  ┌──────▼─────────────────────────▼─────────────────┐   │
│  │         Brute-Force Search Engine                │   │
│  │  ┌──────────────┐       ┌──────────────┐         │   │
│  │  │  Q8 Cosine   │       │  Top-K       │         │   │
│  │  │  Similarity  │       │  Heap        │         │   │
│  │  └──────────────┘       └──────────────┘         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Joplin Data API (userData)
                         ▼
         ┌───────────────────────────────────┐
         │     Per-Note Hidden Storage       │
         │  (invisible, synced properties)   │
         │                                   │
         │  ┌─────────────────────────────┐  │
         │  │  jarvis/v1/meta             │  │
         │  │  jarvis/v1/emb/<model>/0    │  │
         │  └─────────────────────────────┘  │
         └───────────────────────────────────┘
                         │
                         │ Joplin Sync
                         ▼
                   Other Devices
```

### Key Components

1. **UserDataEmbStore**: Manages per-note storage via Joplin's userData API
2. **Embedding Indexer**: Converts note content → embeddings → quantized shards
3. **In-Memory Cache**: Holds all Q8 vectors in RAM for fast brute-force search
4. **Brute-Force Search**: Scans all vectors with Q8 cosine similarity
5. **Top-K Heap**: Maintains best matches during search
6. **Catalog Note**: Stores model registry metadata

---

## 2. Storage Design

### Storage Hierarchy

```
Jarvis Database (Folder)
├── Jarvis Database Catalog (Note)
│   └── userData:
│       ├── jarvis/v1/registry/models → {"<modelId>": true}
│       └── jarvis/v1/models/<modelId>/metadata → CatalogMetadata
│
└── User Notes (any note in Joplin)
    └── userData:
        ├── jarvis/v1/meta → NoteEmbMeta
        └── jarvis/v1/emb/<modelId>/live/0 → EmbShard
```

### Storage Keys

| Key Pattern | Purpose | Scope |
|------------|---------|-------|
| `jarvis/v1/meta` | Multi-model metadata | Per note |
| `jarvis/v1/emb/<modelId>/live/0` | Q8 embedding shard | Per note per model |
| `jarvis/v1/registry/models` | Model registry | Catalog note |
| `jarvis/v1/models/<modelId>/metadata` | Model statistics (noteCount, rowCount, etc.) | Catalog note |

### Single-Shard Constraint

**Design Decision**: Each note stores **exactly one shard** per model (always at index 0).

**Rationale**:
- Simplifies read path (no iteration needed)
- Most notes fit within 500KB limit (~220 blocks for 1536-dim embeddings)
- Prevents userData bloat and sync performance issues
- Truncates large notes gracefully with warnings

**Implications**:
- Notes exceeding the cap have oldest blocks truncated
- Cache key can be simplified: `${noteId}:${modelId}:${epoch}`
- Reads are faster: single userData fetch per note

---

## 3. Data Schemas

### 3.1 NoteEmbMeta (Per-Note Metadata)

**Location**: `jarvis/v1/meta` on each indexed note  
**Purpose**: Tracks which models have embeddings and their versions

```typescript
interface NoteEmbMeta {
  models: {
    [modelId: string]: ModelMetadata
  };
}

interface ModelMetadata {
  dim: number;                      // Embedding dimension (e.g., 512, 1536)
  modelVersion: string;             // Model version string (e.g., "v4.0")
  embeddingVersion: number;         // Embedding format version
  settings: EmbeddingSettings;      // Settings used for this model
  epoch: number;                    // Increments on each update (monotonic)
  contentHash: string;              // MD5 of normalized note content
  shards: number;                   // Number of shards (always 1 currently)
  updatedAt: string;                // ISO8601 timestamp
}

interface EmbeddingSettings {
  embedTitle: boolean;              // Include note title in chunks
  embedPath: boolean;               // Include heading path in chunks
  embedHeading: boolean;            // Include last heading in chunks
  embedTags: boolean;               // Include tags in chunks
  includeCode: boolean;             // Include code blocks
  minLength: number;                // Min block length (chars)
  maxTokens: number;                // Max block size (tokens)
}
```

**Example**:
```json
{
  "models": {
    "Universal Sentence Encoder": {
      "dim": 512,
      "modelVersion": "v4.0",
      "embeddingVersion": 0,
      "settings": {
        "embedTitle": true,
        "embedPath": true,
        "embedHeading": true,
        "embedTags": true,
        "includeCode": false,
        "minLength": 100,
        "maxTokens": 512
      },
      "epoch": 3,
      "contentHash": "d41d8cd98f00b204e9800998ecf8427e",
      "shards": 1,
      "updatedAt": "2025-11-07T10:30:00.000Z"
    }
  }
}
```

---

### 3.2 EmbShard (Per-Note Embeddings)

**Location**: `jarvis/v1/emb/<modelId>/live/0` on each indexed note
**Purpose**: Stores quantized embeddings and block metadata

```typescript
interface EmbShard {
  epoch: number;                    // Must match meta.models[modelId].epoch
  vectorsB64: string;               // Base64-encoded Int8Array (Q8 quantized)
  scalesB64: string;                // Base64-encoded Float32Array (per-row scales)
  meta: BlockRowMeta[];             // Block metadata
}

interface BlockRowMeta {
  title: string;                    // Note title
  headingLevel: number;             // Heading depth (0 = no heading)
  bodyStart: number;                // Character offset in note
  bodyLength: number;               // Length of block text
  lineNumber: number;               // Line number in note
  headingPath?: string[];           // Breadcrumb of headings
}
```

**Example**:
```json
{
  "epoch": 3,
  "vectorsB64": "AQIDBAUGBwgJ...",
  "scalesB64": "Q5oaPUA7wjtA...",
  "meta": [
    {
      "title": "My Research Note",
      "headingLevel": 2,
      "bodyStart": 512,
      "bodyLength": 384,
      "lineNumber": 15,
      "headingPath": ["Introduction", "Background"]
    }
  ]
}
```

---

### 3.3 CatalogMetadata (Per-Model Statistics)

**Location**: `jarvis/v1/models/<modelId>/metadata` on catalog note
**Purpose**: Tracks corpus-wide statistics for each model

```typescript
interface CatalogMetadata {
  modelId: string;                  // Model identifier
  dim: number;                      // Embedding dimension
  version?: string;                 // Model version
  updatedAt?: string;               // Last metadata update
  noteCount?: number;               // Number of notes with embeddings
  rowCount?: number;                // Total embedding blocks across all notes
}
```

**Example**:
```json
{
  "modelId": "Universal Sentence Encoder",
  "dim": 512,
  "version": "v4.0",
  "updatedAt": "2025-11-27T10:00:00.000Z",
  "noteCount": 500,
  "rowCount": 4200
}
```

---

## 4. Algorithms

### 4.1 Block Segmentation

**Algorithm**: Markdown-aware splitter with heading boundaries

**Process**:
1. Parse markdown into sections (headings, paragraphs, lists)
2. Accumulate tokens until `maxTokens` reached
3. Split at heading or paragraph boundaries
4. Optionally include title, heading path, and tags in each block
5. Filter blocks shorter than `minLength` characters

**Metadata Captured**:
- Title, heading level, heading path
- Character offset and length
- Line number in source note

---

### 4.2 Quantization (Q8)

**Algorithm**: Per-row scalar quantization to 8-bit integers

**Forward Quantization**:
```typescript
// For each Float32 vector:
maxAbs = max(|vec[i]|) for i in [0, dim)
scale = maxAbs / 127
q8[i] = clamp(round(vec[i] / scale), -127, 127)
```

**Reconstruction** (used for scoring):
```typescript
// Approximate reconstruction:
float[i] ≈ q8[i] * scale
```

**Cosine Similarity** (Q8 optimized):
```typescript
dot = sum(q8_row[i] * q8_query[i]) for i in [0, dim)
similarity = dot * scale_row * scale_query
```

**Properties**:
- **Compression**: 4× reduction (Float32 → Int8)
- **Accuracy**: ~99% correlation with Float32 cosine similarity
- **Speed**: Integer dot product is faster than FP32
- **Memory**: Stores scale per row (4 bytes overhead per vector)

---

### 4.3 In-Memory Cache

**Purpose**: Hold all Q8 vectors in RAM for fast brute-force search

**Data Structure**:
```typescript
class SimpleCorpusCache {
  vectors: Int8Array;           // Contiguous Q8 vectors (rows × dim)
  scales: Float32Array;         // Per-row scales
  noteIds: string[];            // Note ID for each row
  meta: BlockRowMeta[];         // Block metadata for each row
  dim: number;                  // Embedding dimension
  rows: number;                 // Total number of rows
}
```

**Build Process**:
```typescript
// Lazy build on first search (not at startup):
for each note with embeddings (excluding folders/tags):
  shard = load_shard(note, modelId)
  append shard.vectors to cache.vectors
  append shard.scales to cache.scales
  append shard.meta to cache.meta
  record noteId for each row
```

**Memory Limits**:
| Platform | Limit | Approx Capacity (1536-dim) |
|----------|-------|---------------------------|
| Mobile   | 100MB | ~65,000 blocks |
| Desktop  | 200MB | ~130,000 blocks |

**Capacity Management**:
- Warning logged at 80% of soft limit
- No hard cap - all eligible notes are loaded for complete search results
- Excludes notes in excluded folders (`notes_exclude_folders` setting)
- Excludes notes with `jarvis-exclude` tag
- Power users can control memory usage by excluding folders they don't need in semantic search

**Cache Invalidation**:
- **Incremental**: On note embed/delete, update single note in cache
- **Full rebuild**: On model switch or excluded folders change
- **Dimension mismatch**: Auto-invalidates if model dimension changes

**Performance**:
- Build time: ~2-5s for 10,000+ blocks (one-time)
- Search time: 10-50ms pure RAM scan
- Precision@10: 100% (identical to Float32 baseline)
- Recall@10: 100% (no approximation errors)

---

## 5. Search Pipeline

### Full Search Flow

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Embed Query │  → Float32[dim]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. Quantize Query          │  → Q8 (Int8[dim] + scale)
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. Ensure Cache Built      │
│  - Lazy build on 1st search │
│  - Check dimension match    │
│  - Load all vectors to RAM  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. Brute-Force Scoring     │
│  - Scan all cached vectors  │
│  - Q8 cosine similarity     │
│  - Push to top-K heap       │
│  - Apply minSimilarity      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  5. Result Aggregation      │
│  - Group by note            │
│  - Aggregate similarity     │
│  - Sort by score            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Return top-K   │
│  Notes/Blocks   │
└─────────────────┘
```

### Search Tuning Parameters

| Parameter | Mobile Default | Desktop Default | Purpose |
|-----------|---------------|-----------------|---------|
| `candidateLimit` | 320-800 | 1536-8192 | Max blocks in top-K heap |
| `memoryLimit` | 100MB | 200MB | Max cache size |
| `minSimilarity` | 0.3 | 0.3 | Score threshold for results |

**Memory Usage**: Soft limits with warnings:
- Mobile: ~100MB target (~65,000 blocks @ 1536-dim)
- Desktop: ~200MB target (~130,000 blocks @ 1536-dim)
- Warning logged at 80%, no hard cap (all notes included)

---

## 6. Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Query embedding | O(1) | Model inference (external API or local) |
| Cache build | O(N) | One-time on startup (~2-5s for 10k blocks) |
| Q8 scoring | O(N × dim) | Integer dot product, ~10-50ms |
| Result aggregation | O(K log K) | Top-K heap operations |

**Performance Example** (10,000 blocks, 1536-dim):
- Cache build: ~3 seconds (one-time)
- Search latency: ~20ms (pure RAM scan)

### Space Complexity

**Per Note (userData)**:
- Metadata: ~1KB (JSON)
- Shard: `(dim + 4) × rows × 1.33` bytes (base64 overhead)
- Example (1536-dim, 10 blocks): ~21KB per note

**In-Memory Cache**:
- Q8 vectors: `rows × dim` bytes (Int8Array)
- Scales: `rows × 4` bytes (Float32Array)
- Metadata: ~200 bytes per row
- Example (10,000 blocks, 1536-dim): ~17MB total

**Catalog Storage**:
- Model registry: ~100 bytes per model
- Model metadata: ~200 bytes per model

---

## 7. Platform-Specific Optimizations

### Desktop

**Advantages**:
- More CPU and memory headroom
- Larger cache capacity (200MB)
- Can handle larger corpora

**Tuning**:
- `memoryLimit`: 200MB
- `candidateLimit`: 1536-8192

**Capacity**:
- ~130,000 blocks @ 1536-dim
- ~260,000 blocks @ 512-dim

### Mobile

**Constraints**:
- Limited memory (~100MB budget)
- Slower CPU
- Battery life concerns

**Tuning**:
- `memoryLimit`: 100MB
- `candidateLimit`: 320-800

**Capacity**:
- ~65,000 blocks @ 1536-dim
- ~130,000 blocks @ 512-dim

**Optimizations**:
- Same brute-force algorithm as desktop
- Smaller cache limit prevents OOM
- Warning at 80% capacity

---

## 8. Implementation Details

### Write Path

**Trigger**: Note save or manual DB update

**Process**:
1. Normalize note content
2. Compute MD5 hash
3. Check if `(modelId, contentHash)` unchanged → skip
4. Segment into blocks
5. Embed blocks (parallel API calls)
6. Quantize to Q8
7. Build shard (truncate if >500KB)
8. Increment epoch
9. Write shard to `jarvis/v1/emb/<modelId>/live/0`
10. Update `jarvis/v1/meta`
11. Update in-memory cache (incremental)

**Two-Phase Commit**:
- Write shards first
- Write metadata last (atomic transition)
- Readers ignore mismatched epochs

### Read Path

**Process**:
1. Ensure in-memory cache is built
2. Embed query
3. Quantize query to Q8
4. Scan all cached vectors with Q8 cosine similarity
5. Maintain top-K heap
6. Aggregate results by note
7. Return results

**Validation**:
- Check `shard.epoch == meta.models[modelId].epoch`
- Verify embedding settings match current settings
- Track mismatches for user notification
- Include mismatched notes in results (don't fail)

### Cache Management

**Invalidation Triggers**:
- Model switch → full rebuild
- Excluded folders change → full rebuild
- Note update → incremental update
- Note delete → remove from cache

**Process**:
- Incremental: Update single note's vectors in cache
- Full rebuild: Clear cache and reload from userData

**Safety**:
- Multi-model coexistence: Only delete current model's shards
- Other models remain intact in userData

---

## 9. Future Enhancements

### Under Consideration

1. **Product quantization**: Further compression beyond Q8
2. **L2 distance**: Support Euclidean similarity metric
3. **Approximate search**: For very large corpora exceeding cache limits

---

## 10. Implementation References

### Core Storage & Data Structures
- `src/notes/userDataStore.ts` - Storage abstraction and EmbStore interface
- `src/notes/blockMeta.ts` - Block metadata construction
- `src/notes/shards.ts` - Shard building and size estimation
- `src/notes/q8.ts` - Q8 quantization and cosine similarity

### Search & Indexing
- `src/notes/embeddings.ts` - Main search engine
- `src/notes/embeddingCache.ts` - In-memory Q8 vector cache
- `src/notes/topK.ts` - Top-K heap for result ranking
- `src/notes/userDataReader.ts` - Shard decoding and loading

### Catalog
- `src/notes/catalog.ts` - Catalog note management
- `src/notes/catalogMetadataStore.ts` - Per-model metadata storage

### Indexing & Updates
- `src/notes/userDataIndexer.ts` - Embedding preparation
- `src/notes/validator.ts` - Metadata validation and mismatch detection

### Model Management
- `src/notes/modelSwitch.ts` - Multi-model switching and coverage estimation
- `src/ux/modelManagement.ts` - Model management UI

### Configuration
- `src/ux/settings.ts` - Settings registration and resolution

