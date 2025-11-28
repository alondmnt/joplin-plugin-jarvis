# Issues

REMEMBER: We need to have a clear and simple self-healing path for any database issue, that includes either running DB update manually, or deleting the catalog and running DB update. (CRITICAL)

## ✅ Fixed

### 1. ~~Duplicated catalog note on init~~
**Status:** FIXED
**Solution:** Added in-memory cache + synchronous lock to prevent race condition from Joplin search index delay
**Files:** `src/notes/catalog.ts`

### 2. ~~No toolbar buttons when userData DB is on~~
**Status:** FIXED
**Root cause:** Toolbar buttons were registered AFTER model initialization. If model loading failed (e.g., on mobile with userData DB), the entire startup would abort before UI registration.
**Solution:** **Reordered initialization** to follow a 3-phase approach:
1. **Phase 1**: Lightweight UI setup (settings, dialogs, panel with stub models) - never fails
2. **Phase 2**: Register ALL commands, menus, and toolbar buttons - guaranteed to complete
3. **Phase 3**: Heavy model/DB loading - can fail without breaking UI
**Benefits:**
- Users always see toolbar buttons and menus, even if models fail to load
- Commands fail gracefully at execution time with clear error messages
- Better separation of concerns (UI vs model initialization)
- Easier debugging - users know plugin is installed
**Files:** `src/index.ts` (lines 64-165)

### 3. ~~"Update DB" button not showing on mobile note toolbar when desktop profile selected~~
**Status:** FIXED - 2025-11-22
**Root cause:** Button visibility was checking `notes_device_profile_effective` instead of actual platform
- User selected "desktop" performance profile (for performance tuning)
- Code incorrectly used this setting to determine button visibility
- Button should show based on **actual platform** (UI/UX concern), not **performance profile** (performance tuning)
**Symptoms:**
- On mobile with "desktop" performance profile selected, button didn't appear
- Made it impossible to trigger DB updates without access to Tools menu
**Solution:**
- Changed check from `notes_device_profile_effective` to `notes_device_platform`
- Button now shows based on actual detected platform, regardless of selected performance profile
- Separates UI/UX concerns (button visibility) from performance tuning (search parameters)
**Files:** `src/index.ts` (line 691)

### 4. ~~Memory efficient vector representation~~
**Status:** FIXED - 2025-11-27
**Solution:** Full in-memory corpus cache for all libraries
  - One big Q8 buffer for all vectors (shared Int8Array, cache-friendly)
  - Simple object metadata (easy to debug)
  - Memory: ~6MB for 10K blocks @ 512-dim Q8 (75% savings vs F32)
  - Search: 10-50ms pure RAM (vs 2000ms+ userData I/O)
  - Limits: 100MB mobile, 200MB desktop (with capacity warnings at 80%)
  - Cache invalidates on note changes and excluded folder setting changes
  - Excludes notes in excluded folders and with exclusion tags from cache

---

## 🔍 Open Issues

(none)

---

## ✅ Recently Fixed

### 11. ~~Panel caret expands AND opens note on mobile/web~~
**Status:** FIXED - 2025-11-27
**Platform:** Mobile app, Web clipper (not reproducible on desktop)
**Root cause:** Click handler in `webview.js` checked for class name only, matching both `<summary>` and `<a>` elements with `jarvis-semantic-note` class
**Solution:** Added `element.tagName === 'A'` check to only trigger navigation for actual anchor elements, allowing the `<summary>` caret to work normally for expand/collapse
**Files:** `src/ux/webview.js` (line 5)

---

## Questions & Answers

### 1. Is the min similarity setting being updated when exiting the settings?

**Answer:** ✅ **YES** - Settings are automatically updated
- **Mechanism:** Joplin's `settings.onChange()` handler triggers on any settings change
- **Update flow:** Settings dialog closes → `onChange` fires → `get_settings()` reloads all values
- **Effect:** Immediate - next search uses new threshold
- **No rebuild needed:** This setting affects search filtering, not embeddings

See: `src/index.ts` (line 817: `settings.onChange`), `src/ux/settings.ts` (line 352)

### 2. Do we load all vectors to memory?

**Answer:** ✅ **YES** - All vectors are loaded into an in-memory cache on first search
- **Cache structure:** Single contiguous Q8 buffer + lightweight metadata objects
- **Memory footprint:** ~1 byte per dimension per block (Q8 quantized)
- **Limits:** 100MB on mobile, 200MB on desktop
- **Rebuild triggers:** Note changes (incremental update), excluded folders setting change (full rebuild)
- **Excluded notes:** Notes in excluded folders or with exclusion tags are filtered out during cache build

See: `src/notes/embeddingCache.ts`

---

## Obsolete (IVF Removed)

The following issues were related to IVF (Inverted File Index) centroid-based search, which was removed in favor of simpler in-memory brute-force search (2025-11-27):

- ~~Issue 5: Centroids not trained / only 3 loaded instead of 1024~~
- ~~Issue 6: Centroid training with only 8 samples instead of 20K~~
- ~~Issue 7: Excessive sampling (96% of corpus) during centroid training~~
- ~~Issue 8: IVF performance validation~~
- ~~Issue 9: Centroid assignment gap for single-note updates~~
- ~~Issue 10: Unassigned centroids after k-means~~
