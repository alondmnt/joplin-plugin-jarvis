# Mobile Setup

Jarvis is fully supported on mobile. Most features work immediately; the notes database requires synced storage.

## Contents

- [Features Without a Database](#features-without-a-database)
- [Synced Storage](#synced-storage)
- [Setup](#setup)

## Features Without a Database

Many Jarvis features work immediately without building a notes database:

- **Chat, Ask, Edit, Auto-complete**: Core AI interactions
- **Annotate**: Title and summary generation
- (In the future) **Research**: Literature review

The notes database enables additional features:

- **Related Notes**: Semantic search
- **Chat with Your Notes**: AI responses informed by your notes
- **Annotate**: Links and tags

You can use Jarvis productively while the database builds in the background, or skip it entirely if you only need direct AI features.

## Synced Storage

By default, Jarvis stores embeddings in a local SQLite database. This is fast but desktop-only.

Enabling "Store in note properties" stores embeddings inside your notes instead:

**Advantages:**
- Works on mobile
- Syncs across devices (each note processed once on the first device)
- Backed up with your notes
- Survives reinstalls
- Supports multiple models (each device can use a different model, or the same one for efficiency)

**Trade-offs:**
- Increases database size (~10% with default settings, varies with model and block size)
- Increases sync time
- Experimental feature

## Setup

### Migrating from Desktop

If you have an existing desktop database, migration is faster:

0. **Recommended**: Backup your Joplin database (Tools → Create backup).
1. **Desktop**: Enable "Store in note properties" in Settings → Jarvis → Related Notes → Advanced
2. **Desktop**: Rebuild the database (Tools → Jarvis → Update Jarvis note DB)
3. **Sync**: Wait for sync to complete
4. **Mobile**: Sync on mobile device
5. **Mobile**: Match your desktop settings in "Jarvis: Related Notes" (model, block size, etc.), including enabling "Store in note properties"

Building on desktop is significantly faster than mobile.

### Mobile Only

1. Enable "Store in note properties" in Settings → Jarvis → Related Notes → Advanced
2. Build the database (Tools → Jarvis → Update Jarvis note DB)
3. Wait for indexing to complete (slower on mobile)
