import joplin from 'api';
import { ContentScriptType, MenuItemLocation, ToolbarButtonLocation } from 'api/types';
import debounce from 'lodash.debounce';
import { annotate_title, annotate_summary, annotate_tags, annotate_links } from './commands/annotate';
import { ask_jarvis, edit_with_jarvis } from './commands/ask';
import { chat_with_jarvis, chat_with_notes } from './commands/chat';
import { find_notes, update_note_db, skip_db_init_dialog } from './commands/notes';
import { research_with_jarvis } from './commands/research';
import { load_embedding_model, load_generation_model } from './models/models';
import type { TextEmbeddingModel, TextGenerationModel } from './models/models';
import { find_nearest_notes, clear_corpus_cache } from './notes/embeddings';
import { ensure_catalog_note, get_catalog_note_id } from './notes/catalog';
import { read_model_metadata } from './notes/catalogMetadataStore';
import { register_panel, update_panel } from './ux/panel';
import { get_settings, register_settings, set_folders, get_model_last_sweep_time, get_model_last_full_sweep_time, GENERATION_SETTING_KEYS, EMBEDDING_SETTING_KEYS } from './ux/settings';
import type { JarvisSettings } from './ux/settings';
import { auto_complete } from './commands/complete';
import { getLogger } from './utils/logger';
import type { ModelCoverageStats, ModelSwitchDecision } from './notes/modelSwitch';
import {
  estimate_model_coverage,
  LOW_COVERAGE_THRESHOLD,
  resolve_embedding_model_id,
  prompt_model_switch_decision,
} from './notes/modelSwitch';
import { open_model_management_dialog } from './ux/modelManagement';
import { getModelStats } from './notes/modelStats';
import { checkCapacityWarning } from './notes/embeddingCache';

const STARTUP_DELAY_SECONDS = 5;
const PANEL_DEBOUNCE_SECONDS = 1;
const REFRESH_ATTEMPT_TIMEOUT_MS = 10 * 60 * 1000;

interface UpdateOptions {
  force?: boolean;
  noteIds?: string[];
  silent?: boolean;
  incrementalSweep?: boolean;
}

interface PluginRuntime {
  log: ReturnType<typeof getLogger>;
  settings: JarvisSettings;
  dialogAsk: string;
  model_switch_dialog: string;
  model_management_dialog: string;
  model_embed: TextEmbeddingModel;
  model_gen: TextGenerationModel;
  panel: string;
  delay_scroll: number;
  delay_db_update: number;
  abort_timeout: number;
  pending_note_ids: Set<string>;
  update_abort_controller: AbortController | null;
  update_start_time: number | null;
  initial_sweep_completed: boolean;
  full_sweep_timer: ReturnType<typeof setInterval> | null;
  suppressModelSwitchRevert: boolean;
  lastSyncTime: number;  // Timestamp of last sync completion (for sweep staleness detection)
}

interface UpdateManager {
  start_update: (options?: UpdateOptions) => Promise<void>;
  is_update_in_progress: () => boolean;
}

joplin.plugins.register({
	onStart: async function() {
    // Phase 1: Lightweight initialization (settings, dialogs, panel)
    // This should never fail and ensures UI is always available
    const partialRuntime = await initialize_runtime_ui();
    
    // Create stub models for Phase 2 (will be replaced in Phase 3)
    const stub_embed = { model: null, initialized: false } as any;
    const stub_gen = { model: null, initialized: false } as any;
    
    // Create panel with stub model - panel must exist before registering commands
    const panel = await joplin.views.panels.create('jarvis.relatedNotes');
    register_panel(panel, partialRuntime.settings, stub_embed);
    
    const runtime: PluginRuntime = {
      ...partialRuntime,
      model_embed: stub_embed,
      model_gen: stub_gen,
      panel: panel,
    } as PluginRuntime;
    
    const updates = create_update_manager(runtime);
    const find_notes_debounce = debounce(find_notes, PANEL_DEBOUNCE_SECONDS * 1000);

    // Phase 2: Register all commands, menus, and UI elements
    // Do this BEFORE loading models so users always see the UI
    await register_content_scripts();
    await register_commands_and_menus(runtime, updates, find_notes_debounce);
    await register_workspace_listeners(runtime, updates, find_notes_debounce);
    await register_settings_handler(runtime, updates, find_notes_debounce);

    // Phase 3: Heavy initialization (models, DB) - can fail without breaking UI
    try {
      await initialize_models_and_db(runtime);

      if (runtime.model_embed.model) {
        find_notes_debounce(runtime.model_embed, runtime.panel);
      }

      // Skip startup validation (would scan all notes, ~1min on mobile)
      // Metadata is kept fresh by daily full sweeps instead (see schedule_full_sweep_timer)
      // Stats are set correctly when cache builds on first search (embeddingCache.ts:183)

      await run_initial_sweep(runtime, updates);
      schedule_full_sweep_timer(runtime, updates);
    } catch (error) {
      console.error('Jarvis: Model/DB initialization failed - UI is available but some features may not work', error);
      
      // Show user-friendly error message
      try {
        await joplin.views.dialogs.showMessageBox(
          'Jarvis: Some features failed to initialize. The plugin UI is available but embedding/chat features may not work. Check the console for details.'
        );
      } catch (dialogError) {
        console.error('Jarvis: Could not show error dialog', dialogError);
      }
    }
	},
});

/**
 * Phase 1: Initialize lightweight UI components (settings, dialogs).
 * This should never fail and ensures UI is always available.
 * Models will be loaded later in Phase 3.
 */
async function initialize_runtime_ui(): Promise<Partial<PluginRuntime>> {
  const log = getLogger();
  await register_settings();
  const settings = await get_settings();

  const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');
  const model_switch_dialog = await joplin.views.dialogs.create('jarvis.modelSwitch');
  await joplin.views.dialogs.addScript(model_switch_dialog, 'ux/view.css');
  const model_management_dialog = await joplin.views.dialogs.create('jarvis.modelManagement');
  await joplin.views.dialogs.addScript(model_management_dialog, 'ux/view.css');
  await joplin.views.dialogs.addScript(model_management_dialog, 'ux/modelManagementDialog.css');

  let delay_scroll = await joplin.settings.value('notes_scroll_delay');
  let delay_db_update = 60 * settings.notes_db_update_delay;
  const abort_timeout = 10;  // minutes

  return {
    log,
    settings,
    dialogAsk,
    model_switch_dialog,
    model_management_dialog,
    delay_scroll,
    delay_db_update,
    abort_timeout,
    pending_note_ids: new Set<string>(),
    update_abort_controller: null,
    update_start_time: null,
    initial_sweep_completed: false,
    full_sweep_timer: null,
    suppressModelSwitchRevert: false,
    lastSyncTime: 0,
  };
}

/**
 * Phase 3: Load models and initialize database.
 * This is the heavy lifting that can fail without breaking the UI.
 */
async function initialize_models_and_db(runtime: PluginRuntime): Promise<void> {
  await wait_ms(STARTUP_DELAY_SECONDS * 1000);

  // Load the actual embedding model (replaces stub)
  const model_embed = await load_embedding_model(runtime.settings);
  runtime.model_embed = model_embed;

  if (runtime.settings.notes_db_in_user_data) {
    try {
      await ensure_catalog_note();
    } catch (error) {
      const msg = String((error as any)?.message ?? error);
      if (!/SQLITE_CONSTRAINT|UNIQUE constraint failed/i.test(msg)) {
        console.debug('Jarvis: pre-init catalog deferred', msg);
      }
    }
  }

  if (await skip_db_init_dialog(model_embed)) {
    runtime.delay_db_update = 0;
  }

  // Update panel with the real model (panel was created in Phase 1)
  register_panel(runtime.panel, runtime.settings, model_embed);

  // Load the actual generation model (replaces stub)
  const model_gen = await load_generation_model(runtime.settings);
  runtime.model_gen = model_gen;
}

/**
 * Create helpers for coordinating background embedding updates.
 */
function create_update_manager(runtime: PluginRuntime): UpdateManager {
  const is_update_in_progress = (): boolean => {
    console.debug('is_update_in_progress', runtime.update_abort_controller, runtime.update_start_time);
    return runtime.update_abort_controller !== null &&
      (runtime.update_start_time !== null && (Date.now() - runtime.update_start_time) < runtime.abort_timeout * 60 * 1000);
  };

  const start_update = async (options: UpdateOptions = {}) => {
    const { force = false, noteIds, silent = false, incrementalSweep = false } = options;
    const targetIds = noteIds && noteIds.length > 0 ? Array.from(new Set(noteIds)) : undefined;
    console.debug('start_update', is_update_in_progress(), force, targetIds?.length ?? 0, 'incremental:', incrementalSweep);

    if (targetIds && targetIds.length === 0 && !force) {
      return;
    }

    if (is_update_in_progress()) {
      if (!force) {
        if (!silent) {
          await joplin.views.dialogs.showMessageBox('Update already in progress');
        }
        return;
      }
      runtime.update_abort_controller?.abort();
    } else if (runtime.update_abort_controller !== null) {
      runtime.update_abort_controller.abort();
    }

    if (runtime.update_abort_controller !== null) {
      runtime.update_abort_controller = null;
    }
    runtime.update_abort_controller = new AbortController();
    runtime.update_start_time = Date.now();

    try {
      await update_note_db(runtime.model_embed, runtime.panel, runtime.update_abort_controller, targetIds, force, incrementalSweep, runtime.lastSyncTime);
    } finally {
      runtime.update_abort_controller = null;
      runtime.update_start_time = null;
    }
  };

  return {
    start_update,
    is_update_in_progress,
  };
}

/**
 * Stop the periodic sweep timer if one is active.
 */
function cancel_full_sweep_timer(runtime: PluginRuntime): void {
  if (runtime.full_sweep_timer !== null) {
    clearInterval(runtime.full_sweep_timer);
    runtime.full_sweep_timer = null;
  }
}

/**
 * Schedule periodic userData sweeps to keep remote changes fresh.
 */
function schedule_full_sweep_timer(runtime: PluginRuntime, updates: UpdateManager): void {
  cancel_full_sweep_timer(runtime);
  if (runtime.delay_db_update <= 0) {
    return;
  }

  runtime.full_sweep_timer = setInterval(() => {
    void (async () => {
      if (runtime.model_embed.model === null) {
        return;
      }
      if (updates.is_update_in_progress()) {
        return;
      }
      try {
        // Determine sweep type: full sweep once per day for self-healing and metadata updates
        const lastSweepTime = await get_model_last_sweep_time(runtime.model_embed.id);
        const lastFullSweepTime = await get_model_last_full_sweep_time(runtime.model_embed.id);
        const now = Date.now();
        const ONE_DAY_MS = 24 * 60 * 60 * 1000;

        // Use full sweep if: never done before OR more than 24h since last full sweep
        const needsFullSweep = lastSweepTime === 0 || (now - lastFullSweepTime) > ONE_DAY_MS;
        const useIncremental = !needsFullSweep;

        if (needsFullSweep) {
          console.info('Jarvis: running daily full sweep for metadata update and self-healing');
        }

        const updateOptions: UpdateOptions = { force: false, silent: true, incrementalSweep: useIncremental };
        await updates.start_update(updateOptions);
        runtime.pending_note_ids.clear();
        runtime.initial_sweep_completed = true;
      } catch (error) {
        console.warn('Jarvis: periodic note DB sweep failed', error);
      }
    })();
  }, runtime.delay_db_update * 1000);
}

/**
 * Run the initial background sweep after startup if requested.
 */
async function run_initial_sweep(runtime: PluginRuntime, updates: UpdateManager): Promise<void> {
  if (runtime.model_embed.model === null) {
    return;
  }

  if (runtime.delay_db_update <= 0) {
    return;
  }

  try {
    // Only use incremental sweep if database has been built before (check settings)
    // First build needs full scan to populate database
    const lastSweepTime = await get_model_last_sweep_time(runtime.model_embed.id);
    const useIncremental = lastSweepTime > 0;

    const updateOptions: UpdateOptions = { force: false, silent: true, incrementalSweep: useIncremental };

    await updates.start_update(updateOptions);
    runtime.pending_note_ids.clear();
    runtime.initial_sweep_completed = true;
  } catch (error) {
    console.warn('Jarvis: initial note DB sweep failed', error);
  }
}

/**
 * Register editor content scripts for scrolling support.
 */
async function register_content_scripts(): Promise<void> {
  await joplin.contentScripts.register(
    ContentScriptType.CodeMirrorPlugin,
    'jarvis.cm5scroller',
    './cm5scroller.js',
  );
  await joplin.contentScripts.register(
    ContentScriptType.CodeMirrorPlugin,
    'jarvis.cm6scroller',
    './cm6scroller.js',
  );
}

/**
 * Register all commands, menus, and toolbar buttons used by the plugin.
 */
async function register_commands_and_menus(
  runtime: PluginRuntime,
  updates: UpdateManager,
  find_notes_debounce: (model: TextEmbeddingModel, panel: string) => void,
): Promise<void> {
  await joplin.commands.register({
    name: 'jarvis.ask',
    label: 'Ask Jarvis',
    execute: async () => {
      await ask_jarvis(runtime.model_gen, runtime.dialogAsk);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.chat',
    label: 'Chat with Jarvis',
    iconName: 'fas fa-robot',
    execute: async () => {
      await chat_with_jarvis(runtime.model_gen);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.research',
    label: 'Research with Jarvis',
    execute: async () => {
      await research_with_jarvis(runtime.model_gen, runtime.dialogAsk);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.edit',
    label: 'Edit selection with Jarvis',
    iconName: 'far fa-edit',
    execute: async () => {
      await edit_with_jarvis(runtime.model_gen, runtime.dialogAsk);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.complete',
    label: 'Auto-complete with Jarvis',
    iconName: 'fas fa-magic',
    execute: async () => {
      await auto_complete(runtime.model_gen, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.annotate.title',
    label: 'Annotate note: title',
    execute: async () => {
      await annotate_title(runtime.model_gen, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.annotate.summary',
    label: 'Annotate note: summary',
    execute: async () => {
      await annotate_summary(runtime.model_gen, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.annotate.tags',
    label: 'Annotate note: tags',
    execute: async () => {
      await annotate_tags(runtime.model_gen, runtime.model_embed, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.annotate.links',
    label: 'Annotate note: links',
    execute: async () => {
      await annotate_links(runtime.model_embed, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.annotate.button',
    label: 'Annotate note with Jarvis',
    iconName: 'fas fa-lightbulb',
    execute: async () => {
      const settings = runtime.settings;
      if (settings.annotate_links_flag) {
        await annotate_links(runtime.model_embed, settings);
      }

      if (settings.annotate_summary_flag || settings.annotate_title_flag || settings.annotate_tags_flag) {
        const summary = await annotate_summary(runtime.model_gen, settings, settings.annotate_summary_flag);
        if (settings.annotate_title_flag) {
          await annotate_title(runtime.model_gen, settings, summary);
        }
        if (settings.annotate_tags_flag) {
          await annotate_tags(runtime.model_gen, runtime.model_embed, settings, summary);
        }
      }
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.db.update',
    label: 'Update Jarvis note DB',
    iconName: 'fas fa-sync-alt',
    execute: async () => {
      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }
      await updates.start_update({ force: true });
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.manage_models',
    label: 'Manage Jarvis note DB',
    execute: async () => {
      await open_model_management_dialog(runtime.model_management_dialog, runtime.panel, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.find',
    label: 'Find related notes',
    iconName: 'fas fa-search',
    execute: async () => {
      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }
      find_notes_debounce(runtime.model_embed, runtime.panel);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.toggle_panel',
    label: 'Toggle related notes panel',
    execute: async () => {
      if (await joplin.views.panels.visible(runtime.panel)) {
        await joplin.views.panels.hide(runtime.panel);
      } else {
        await joplin.views.panels.show(runtime.panel);
        if (runtime.model_embed.model === null) {
          await runtime.model_embed.initialize();
        }
        find_notes_debounce(runtime.model_embed, runtime.panel);
      }
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.chat',
    label: 'Chat with your notes',
    iconName: 'fas fa-comments',
    execute: async () => {
      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }
      await chat_with_notes(runtime.model_embed, runtime.model_gen, runtime.panel);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.preview',
    label: 'Preview chat notes context',
    execute: async () => {
      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }
      await chat_with_notes(runtime.model_embed, runtime.model_gen, runtime.panel, true);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.utils.count_tokens',
    label: 'Count tokens in selection',
    execute: async () => {
      const text = await joplin.commands.execute('selectedText');
      const tokenCount = runtime.model_gen.count_tokens(text);
      await joplin.views.dialogs.showMessageBox(`Token count: ${tokenCount}`);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.exclude_folder',
    label: 'Exclude notebook from note DB',
    execute: async () => {
      const folder = await joplin.workspace.selectedFolder();
      if (!folder) {
        return;
      }
      set_folders(true, folder.id, runtime.settings);
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.include_folder',
    label: 'Include notebook in note DB',
    execute: async () => {
      const folder = await joplin.workspace.selectedFolder();
      if (!folder) {
        return;
      }
      set_folders(false, folder.id, runtime.settings);
    },
  });

  joplin.views.menus.create('jarvis', 'Jarvis', [
    { commandName: 'jarvis.chat', accelerator: 'CmdOrCtrl+Shift+C' },
    { commandName: 'jarvis.notes.chat', accelerator: 'CmdOrCtrl+Alt+C' },
    { commandName: 'jarvis.ask', accelerator: 'CmdOrCtrl+Shift+J' },
    { commandName: 'jarvis.research', accelerator: 'CmdOrCtrl+Shift+R' },
    { commandName: 'jarvis.edit', accelerator: 'CmdOrCtrl+Shift+E' },
    { commandName: 'jarvis.complete', accelerator: 'CmdOrCtrl+Shift+A' },
    { commandName: 'jarvis.annotate.title' },
    { commandName: 'jarvis.annotate.summary' },
    { commandName: 'jarvis.annotate.links' },
    { commandName: 'jarvis.annotate.tags' },
    { commandName: 'jarvis.notes.find', accelerator: 'CmdOrCtrl+Alt+F' },
    { commandName: 'jarvis.notes.preview' },
    { commandName: 'jarvis.utils.count_tokens' },
    { commandName: 'jarvis.notes.db.update' },
    { commandName: 'jarvis.notes.manage_models' },
    { commandName: 'jarvis.notes.toggle_panel' },
    { commandName: 'jarvis.notes.exclude_folder' },
    { commandName: 'jarvis.notes.include_folder' },
  ], MenuItemLocation.Tools);

  const toolbarSettingKeys = [
    'toolbar_show_chat',
    'toolbar_show_notes_chat',
    'toolbar_show_notes_find',
    'toolbar_show_edit',
    'toolbar_show_complete',
    'toolbar_show_annotate',
  ];
  const toolbarSettings = await joplin.settings.values(toolbarSettingKeys);
  const toolbarButtons = [
    { id: 'jarvis.toolbar.chat', command: 'jarvis.chat', enabled: toolbarSettings.toolbar_show_chat },
    { id: 'jarvis.toolbar.notes.chat', command: 'jarvis.notes.chat', enabled: toolbarSettings.toolbar_show_notes_chat },
    { id: 'jarvis.toolbar.notes.find', command: 'jarvis.notes.find', enabled: toolbarSettings.toolbar_show_notes_find },
    { id: 'jarvis.toolbar.edit', command: 'jarvis.edit', enabled: toolbarSettings.toolbar_show_edit },
    { id: 'jarvis.toolbar.complete', command: 'jarvis.complete', enabled: toolbarSettings.toolbar_show_complete },
    { id: 'jarvis.toolbar.annotate', command: 'jarvis.annotate.button', enabled: toolbarSettings.toolbar_show_annotate },
  ];
  for (const button of toolbarButtons) {
    if (!button.enabled) {
      continue;
    }
    await joplin.views.toolbarButtons.create(button.id, button.command, ToolbarButtonLocation.EditorToolbar);
  }

  // On mobile, add "Update Jarvis note DB" and "Toggle panel" buttons to NoteToolbar
  // This is needed because imported notes with old timestamps won't be caught by incremental sweeps
  // and mobile doesn't have easy access to the Tools menu or panels
  // Note: Use actual platform detection, not effective profile (which is for performance tuning)
  if (runtime.settings.notes_device_platform === 'mobile') {
    await joplin.views.toolbarButtons.create(
      'jarvis.toolbar.notes.db.update',
      'jarvis.notes.db.update',
      ToolbarButtonLocation.NoteToolbar
    );
    await joplin.views.toolbarButtons.create(
      'jarvis.toolbar.notes.toggle_panel',
      'jarvis.notes.toggle_panel',
      ToolbarButtonLocation.NoteToolbar
    );
  }

  await joplin.views.menuItems.create('jarvis.context.notes.find', 'jarvis.notes.find', MenuItemLocation.EditorContextMenu);
  await joplin.views.menuItems.create('jarvis.context.utils.count_tokens', 'jarvis.utils.count_tokens', MenuItemLocation.EditorContextMenu);
  await joplin.views.menuItems.create('jarvis.context.edit', 'jarvis.edit', MenuItemLocation.EditorContextMenu);
}

/**
 * Register workspace-level listeners for note changes and panel messages.
 */
async function register_workspace_listeners(
  runtime: PluginRuntime,
  updates: UpdateManager,
  find_notes_debounce: (model: TextEmbeddingModel, panel: string) => void,
): Promise<void> {
  const flush_note_changes = async (options: { silent?: boolean } = {}) => {
    if (runtime.pending_note_ids.size === 0) {
      return;
    }

    if (updates.is_update_in_progress()) {
      return;
    }

    if (runtime.model_embed.model === null) {
      return;
    }

    const noteIds = Array.from(runtime.pending_note_ids);
    runtime.pending_note_ids.clear();
    try {
      await updates.start_update({ noteIds, silent: options.silent ?? true });
    } catch (error) {
      for (const id of noteIds) {
        runtime.pending_note_ids.add(id);
      }
      throw error;
    }
  };

  await joplin.workspace.onNoteSelectionChange(async () => {
    if (runtime.model_embed.model === null) {
      await runtime.model_embed.initialize();
    }
    if (runtime.pending_note_ids.size > 0) {
    void flush_note_changes({ silent: true }).catch((error) => {
        console.warn('Jarvis: incremental note update failed', error);
      });
    }
    if (!runtime.initial_sweep_completed && runtime.delay_db_update > 0 && runtime.model_embed.model !== null) {
      void (async () => {
        if (updates.is_update_in_progress()) {
          return;
        }
        try {
          // Check if userData exists on other devices but hasn't synced locally yet
          // If catalog shows data but we have no local data, wait for sync
          if (runtime.settings.notes_db_in_user_data) {
            const catalogId = await get_catalog_note_id();
            if (catalogId) {
              const catalogMeta = await read_model_metadata(catalogId, runtime.model_embed.id);
              const hasRemoteData = catalogMeta?.noteCount > 0;
              const hasLocalData = runtime.model_embed.embeddings.length > 0;
              if (hasRemoteData && !hasLocalData) {
                console.info('Jarvis: userData exists on other devices, waiting for sync before sweep', {
                  modelId: runtime.model_embed.id,
                  remoteNoteCount: catalogMeta?.noteCount,
                });
                return; // Skip this sweep, let sync complete first
              }
            }
          }

          // Only use incremental sweep if database has been built before (check settings)
          // Force full sweep if migration is needed (SQLite has data but userData not built yet)
          const lastSweepTime = await get_model_last_sweep_time(runtime.model_embed.id);
          const needsMigration = runtime.settings.notes_db_in_user_data
            && !runtime.settings.notes_model_first_build_completed?.[runtime.model_embed.id]
            && runtime.model_embed.embeddings.length > 0;
          const useIncremental = lastSweepTime > 0 && !needsMigration;
          if (needsMigration) {
            console.info('Jarvis: migration needed - forcing full sweep to backfill SQLite → userData');
          }
          await updates.start_update({ force: false, silent: true, incrementalSweep: useIncremental });
          runtime.pending_note_ids.clear();
          runtime.initial_sweep_completed = true;
        } catch (error) {
          console.warn('Jarvis: initial note DB sweep retry failed', error);
        }
      })();
    }
    find_notes_debounce(runtime.model_embed, runtime.panel);
  });

  await joplin.workspace.onNoteChange(async (event: any) => {
    const noteId = event?.id ?? event?.item?.id;
    if (!noteId) {
      return;
    }
    runtime.pending_note_ids.add(noteId);
  });

  await joplin.workspace.onSyncComplete(async () => {
    runtime.lastSyncTime = Date.now();
    console.debug('Jarvis: sync completed at', new Date(runtime.lastSyncTime).toISOString());
  });

  await joplin.views.panels.onMessage(runtime.panel, async (message) => {
    if (message.name === 'openRelatedNote') {
      // Dismiss plugin panels first (required for web/mobile to allow note opening)
      try {
        await joplin.commands.execute('dismissPluginPanels');
      } catch {
        // Ignore errors (not on mobile/web, or old version)
      }
      await joplin.commands.execute('openNote', message.note);
      if (message.line > 0) {
        await wait_ms(runtime.delay_scroll);
        await joplin.commands.execute('editor.execCommand', {
          name: 'scrollToJarvisLine',
          args: [message.line - 1],
        });
      }
    }
    if (message.name === 'searchRelatedNote') {
      const nearest = await find_nearest_notes(
        runtime.model_embed.embeddings,
        '1234',
        1,
        '',
        message.query,
        runtime.model_embed,
        runtime.settings,
        true,
        runtime.panel,
        updates.is_update_in_progress()
      );
      // Compute capacity warning from in-memory stats (if available)
      const stats = getModelStats(runtime.model_embed.id);
      const profileIsDesktop = runtime.settings.notes_device_profile_effective === 'desktop';
      const capacityWarning = stats
        ? checkCapacityWarning(stats.rowCount, stats.dim, profileIsDesktop)
        : null;
      await update_panel(runtime.panel, nearest, runtime.settings, capacityWarning);
    }
    if (message.name === 'abortUpdate') {
      runtime.update_abort_controller?.abort();
    }
  });
}

/**
 * Register handler for settings changes (model reloads, coverage checks).
 */
async function register_settings_handler(
  runtime: PluginRuntime,
  updates: UpdateManager,
  find_notes_debounce: (model: TextEmbeddingModel, panel: string) => void,
): Promise<void> {
  await joplin.settings.onChange(async (event) => {
    const previousSettings = runtime.settings;

    if (runtime.suppressModelSwitchRevert) {
      runtime.suppressModelSwitchRevert = false;
      runtime.settings = await get_settings();
      return;
    }

    runtime.settings = await get_settings();

    const reloadGeneration = event.keys.some((key: string) => GENERATION_SETTING_KEYS.has(key));
    const reloadEmbedding = event.keys.some((key: string) => EMBEDDING_SETTING_KEYS.has(key));

    const notesModelChanged = event.keys.includes('notes_model')
      && previousSettings?.notes_model !== runtime.settings.notes_model;

    // Check if excluded folders changed - invalidate cache to reload with new filter
    const excludedFoldersChanged = event.keys.includes('notes_exclude_folders');
    if (excludedFoldersChanged && runtime.settings.notes_db_in_user_data) {
      const modelId = resolve_embedding_model_id(runtime.settings);
      if (modelId) {
        clear_corpus_cache(modelId);
        runtime.log.info('Jarvis: invalidated cache due to excluded folders change');
      }
    }

    let forceUpdateAfterReload = false;
    let skipUpdateAfterReload = false;

    if (notesModelChanged) {
      const newModelId = resolve_embedding_model_id(runtime.settings);
      const oldModelId = resolve_embedding_model_id(previousSettings);
      
      // Clear old model's in-memory cache to free memory
      if (oldModelId && oldModelId !== newModelId) {
        clear_corpus_cache(oldModelId);
      }

      if (runtime.settings.notes_db_in_user_data && newModelId) {
        // Skip coverage check if migration from SQLite may be pending
        // (firstBuildCompleted is false AND model was used before, indicated by lastSweepTime > 0)
        const firstBuildCompleted = Boolean(runtime.settings.notes_model_first_build_completed?.[newModelId]);
        const lastSweepTime = await get_model_last_sweep_time(newModelId);
        const migrationPending = !firstBuildCompleted && lastSweepTime > 0;

        if (migrationPending) {
          runtime.log.info('Jarvis: skipping coverage check for model (migration pending)', { modelId: newModelId, lastSweepTime });
        }

        let coverageStats: ModelCoverageStats | null = null;
        if (!migrationPending) {
          try {
            coverageStats = await estimate_model_coverage(newModelId);
          } catch (error) {
            runtime.log.warn('Jarvis: failed to estimate model coverage', { modelId: newModelId, error });
          }
        }

        const totalNotes = coverageStats?.totalNotes ?? 0;
        const coverageRatio = coverageStats?.coverageRatio ?? 1;

        if (coverageStats && totalNotes > 0 && coverageRatio < LOW_COVERAGE_THRESHOLD) {
          const decision: ModelSwitchDecision = await prompt_model_switch_decision(runtime.model_switch_dialog, coverageStats, newModelId);
          if (decision === 'cancel') {
            runtime.suppressModelSwitchRevert = true;
            await joplin.settings.setValue('notes_model', previousSettings?.notes_model ?? '');
            runtime.settings = previousSettings;
            runtime.log.info('Jarvis: model switch cancelled by user', {
              from: oldModelId,
              to: newModelId,
              coverage: coverageRatio,
            });
            return;
          }

          if (decision === 'populate') {
            forceUpdateAfterReload = true;
            runtime.log.info('Jarvis: low coverage populate selected', {
              from: oldModelId,
              to: newModelId,
              coverage: coverageRatio,
              sampled: coverageStats.sampledNotes,
              estimatedNotes: coverageStats.estimatedNotesWithModel,
            });
          } else {
            skipUpdateAfterReload = true;
            runtime.log.warn('Jarvis: low coverage switch without populate', {
              from: oldModelId,
              to: newModelId,
              coverage: coverageRatio,
              sampled: coverageStats.sampledNotes,
              estimatedNotes: coverageStats.estimatedNotesWithModel,
            });
          }
        } else {
          forceUpdateAfterReload = true;
          if (coverageStats) {
            runtime.log.info('Jarvis: model switch coverage OK', {
              from: oldModelId,
              to: newModelId,
              coverage: coverageRatio,
              sampled: coverageStats.sampledNotes,
              estimatedNotes: coverageStats.estimatedNotesWithModel,
            });
          } else {
            runtime.log.warn('Jarvis: model switch coverage unavailable, defaulting to populate', {
              from: oldModelId,
              to: newModelId,
            });
          }
        }
      } else {
        forceUpdateAfterReload = true;
        runtime.log.info('Jarvis: model switch without userData coverage check', {
          from: oldModelId,
          to: newModelId,
        });
      }
    }

    if (reloadGeneration) {
      runtime.model_gen = await load_generation_model(runtime.settings);
    }

    if (reloadEmbedding) {
      runtime.model_embed = await load_embedding_model(runtime.settings);

      if (runtime.settings.notes_db_in_user_data) {
        try {
          await ensure_catalog_note();
        } catch (error) {
          const msg = String((error as any)?.message ?? error);
          if (!/SQLITE_CONSTRAINT|UNIQUE constraint failed/i.test(msg)) {
            console.debug('Jarvis: pre-init catalog deferred (settings change)', msg);
          }
        }
      }

      if (runtime.model_embed.model && !skipUpdateAfterReload) {
        try {
          // Only use incremental sweep if database has been built before (check settings)
          const lastSweepTime = await get_model_last_sweep_time(runtime.model_embed.id);
          const useIncremental = lastSweepTime > 0;
          const updateOptions = forceUpdateAfterReload
            ? { force: true }  // Model change needs full validation
            : { force: false, silent: true, incrementalSweep: useIncremental };
          await updates.start_update(updateOptions);
          runtime.pending_note_ids.clear();
          runtime.initial_sweep_completed = true;
        } catch (error) {
          console.warn('Jarvis: settings-triggered note DB sweep failed', error);
        }
      } else if (skipUpdateAfterReload) {
        runtime.log.info('Jarvis: skipped automatic populate for new model', {
          modelId: runtime.model_embed?.id,
        });
      }
    }

    if (event.keys.includes('notes_scroll_delay')) {
      runtime.delay_scroll = await joplin.settings.value('notes_scroll_delay');
    }

    if (runtime.model_embed.model) {
      find_notes_debounce(runtime.model_embed, runtime.panel);
    }

    if (event.keys.includes('notes_db_update_delay')) {
      runtime.delay_db_update = 60 * runtime.settings.notes_db_update_delay;
      schedule_full_sweep_timer(runtime, updates);
      if (runtime.delay_db_update > 0 && runtime.model_embed.model !== null) {
        void (async () => {
        if (updates.is_update_in_progress()) {
            return;
          }
          try {
            // Check if userData exists on other devices but hasn't synced locally yet
            if (runtime.settings.notes_db_in_user_data) {
              const catalogId = await get_catalog_note_id();
              if (catalogId) {
                const catalogMeta = await read_model_metadata(catalogId, runtime.model_embed.id);
                const hasRemoteData = catalogMeta?.noteCount > 0;
                const hasLocalData = runtime.model_embed.embeddings.length > 0;
                if (hasRemoteData && !hasLocalData) {
                  console.info('Jarvis: userData exists on other devices, waiting for sync before sweep');
                  return;
                }
              }
            }

            // Only use incremental sweep if database has been built before (check settings)
            // Force full sweep if migration is needed (SQLite has data but userData not built yet)
            const lastSweepTime = await get_model_last_sweep_time(runtime.model_embed.id);
            const needsMigration = runtime.settings.notes_db_in_user_data
              && !runtime.settings.notes_model_first_build_completed?.[runtime.model_embed.id]
              && runtime.model_embed.embeddings.length > 0;
            const useIncremental = lastSweepTime > 0 && !needsMigration;
            if (needsMigration) {
              console.info('Jarvis: migration needed - forcing full sweep to backfill SQLite → userData');
            }
            await updates.start_update({ force: false, silent: true, incrementalSweep: useIncremental });
            runtime.pending_note_ids.clear();
            runtime.initial_sweep_completed = true;
          } catch (error) {
            console.warn('Jarvis: note DB sweep failed after delay change', error);
          }
        })();
      }
    }
  });
}

/**
 * Sleep helper (milliseconds).
 */
function wait_ms(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
 
