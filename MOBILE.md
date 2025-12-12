# Mobile Setup

Jarvis supports mobile devices when using synced storage.

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
1. **Desktop**: Enable "Store in note properties" in Settings → Jarvis → Related Notes
2. **Desktop**: Rebuild the database (Tools → Jarvis → Update Jarvis note DB)
3. **Sync**: Wait for sync to complete
4. **Mobile**: Sync on mobile device
5. **Mobile**: Match your desktop settings in "Jarvis: Related Notes" (model, block size, etc.), including enabling "Store in note properties"

Building on desktop is significantly faster than mobile.

### Mobile Only

1. Enable "Store in note properties" in Settings → Jarvis → Related Notes
2. Build the database (Tools → Jarvis → Update Jarvis note DB)
3. Wait for indexing to complete (slower on mobile)
