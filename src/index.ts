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
import { find_nearest_notes } from './notes/embeddings';
import { ensure_catalog_note, get_catalog_note_id, resolve_anchor_note_id } from './notes/catalog';
import { register_panel, update_panel } from './ux/panel';
import { get_settings, register_settings, set_folders, GENERATION_SETTING_KEYS, EMBEDDING_SETTING_KEYS } from './ux/settings';
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
import { read_anchor_meta_data, write_anchor_metadata, AnchorRefreshState } from './notes/anchorStore';

const STARTUP_DELAY_SECONDS = 5;
const PANEL_DEBOUNCE_SECONDS = 1;
const REFRESH_ATTEMPT_TIMEOUT_MS = 10 * 60 * 1000;

interface UpdateOptions {
  force?: boolean;
  noteIds?: string[];
  silent?: boolean;
}

interface PluginRuntime {
  log: ReturnType<typeof getLogger>;
  settings: JarvisSettings;
  dialogAsk: string;
  model_switch_dialog: string;
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
}

interface UpdateManager {
  start_update: (options?: UpdateOptions) => Promise<void>;
  is_update_in_progress: () => boolean;
}

joplin.plugins.register({
	onStart: async function() {
    const runtime = await initialize_runtime();
    const updates = create_update_manager(runtime);
    const find_notes_debounce = debounce(find_notes, PANEL_DEBOUNCE_SECONDS * 1000);

    if (runtime.model_embed.model) {
      find_notes_debounce(runtime.model_embed, runtime.panel);
    }

    await run_initial_sweep(runtime, updates);
    schedule_full_sweep_timer(runtime, updates);

    await register_content_scripts();
    await register_commands_and_menus(runtime, updates, find_notes_debounce);
    await register_workspace_listeners(runtime, updates, find_notes_debounce);
    await register_settings_handler(runtime, updates, find_notes_debounce);
	},
});

/**
 * Initialize runtime state, load models, dialogs, and UI components.
 */
async function initialize_runtime(): Promise<PluginRuntime> {
  const log = getLogger();
  await register_settings();
  const settings = await get_settings();

  const dialogAsk = await joplin.views.dialogs.create('jarvis.ask.dialog');
  const model_switch_dialog = await joplin.views.dialogs.create('jarvis.modelSwitch');
  await joplin.views.dialogs.addScript(model_switch_dialog, 'ux/view.css');

  let delay_scroll = await joplin.settings.value('notes_scroll_delay');
  let delay_db_update = 60 * settings.notes_db_update_delay;
  const abort_timeout = 10;  // minutes

  await wait_ms(STARTUP_DELAY_SECONDS * 1000);

  const model_embed = await load_embedding_model(settings);

  if (settings.experimental_user_data_index) {
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
    delay_db_update = 0;
  }

  const panel = await joplin.views.panels.create('jarvis.relatedNotes');
  register_panel(panel, settings, model_embed);

  const model_gen = await load_generation_model(settings);

  return {
    log,
    settings,
    dialogAsk,
    model_switch_dialog,
    model_embed,
    model_gen,
    panel,
    delay_scroll,
    delay_db_update,
    abort_timeout,
    pending_note_ids: new Set<string>(),
    update_abort_controller: null,
    update_start_time: null,
    initial_sweep_completed: false,
    full_sweep_timer: null,
    suppressModelSwitchRevert: false,
  };
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
    const { force = false, noteIds, silent = false } = options;
    const targetIds = noteIds && noteIds.length > 0 ? Array.from(new Set(noteIds)) : undefined;
    console.debug('start_update', is_update_in_progress(), force, targetIds?.length ?? 0);

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
      await update_note_db(runtime.model_embed, runtime.panel, runtime.update_abort_controller, targetIds, force);
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

async function claim_centroid_refresh(runtime: PluginRuntime): Promise<boolean> {
  if (!runtime.settings.experimental_user_data_index) {
    return false;
  }
  if (runtime.settings.notes_device_profile_effective === 'mobile') {
    return false;
  }

  const modelId = resolve_embedding_model_id(runtime.settings);
  if (!modelId) {
    return false;
  }

  try {
    const catalogId = await get_catalog_note_id();
    if (!catalogId) {
      return false;
    }

    const anchorId = await resolve_anchor_note_id(catalogId, modelId);
    if (!anchorId) {
      return false;
    }

    const metadata = await read_anchor_meta_data(anchorId);
    const refresh = metadata?.refresh;
    if (!metadata || !refresh) {
      return false;
    }

    const now = Date.now();
    const lastAttemptMs = Date.parse(refresh.lastAttemptAt ?? refresh.requestedAt ?? '') || 0;
    const attemptStale = refresh.status === 'in_progress'
      && (now - lastAttemptMs) > REFRESH_ATTEMPT_TIMEOUT_MS;

    if (refresh.status !== 'pending' && !attemptStale) {
      return false;
    }

    const updatedRefresh: AnchorRefreshState = {
      ...refresh,
      status: 'in_progress',
      lastAttemptAt: new Date().toISOString(),
    };

    await write_anchor_metadata(anchorId, {
      ...metadata,
      refresh: updatedRefresh,
    });

    runtime.log.info('Jarvis: claimed centroid refresh request', {
      modelId,
      reason: refresh.reason,
      requestedAt: refresh.requestedAt,
    });

    return true;
  } catch (error) {
    runtime.log.warn('Jarvis: failed to claim centroid refresh request', error);
    return false;
  }
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
        const refreshClaimed = await claim_centroid_refresh(runtime);
        const updateOptions: UpdateOptions = refreshClaimed
          ? { force: true }
          : { force: false, silent: true };
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

  try {
    const refreshClaimed = await claim_centroid_refresh(runtime);
    if (!refreshClaimed && runtime.delay_db_update <= 0) {
      return;
    }

    const updateOptions: UpdateOptions = refreshClaimed
      ? { force: true }
      : { force: false, silent: true };

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
    './content_scripts/cm5scroller.js',
  );
  await joplin.contentScripts.register(
    ContentScriptType.CodeMirrorPlugin,
    'jarvis.cm6scroller',
    './content_scripts/cm6scroller.js',
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
    execute: async () => {
      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }
      await updates.start_update({ force: true });
    },
  });

  await joplin.commands.register({
    name: 'jarvis.notes.db.updateSubset',
    execute: async (args?: any) => {
      const noteIds: string[] = Array.isArray(args)
        ? args
        : Array.isArray(args?.noteIds)
          ? args.noteIds
          : [];

      if (!noteIds || noteIds.length === 0) {
        runtime.log.info('Validation rebuild requested with no note IDs, skipping');
        return;
      }

      const uniqueIds = Array.from(new Set(noteIds));

      if (runtime.model_embed.model === null) {
        await runtime.model_embed.initialize();
      }

      runtime.log.info(`Validation subset rebuild triggered for ${uniqueIds.length} notes`);
      await updates.start_update({ force: true, noteIds: uniqueIds });
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
          await updates.start_update({ force: false, silent: true });
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

  await joplin.views.panels.onMessage(runtime.panel, async (message) => {
    if (message.name === 'openRelatedNote') {
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
      );
      await update_panel(runtime.panel, nearest, runtime.settings);
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

    let forceUpdateAfterReload = false;
    let skipUpdateAfterReload = false;

    if (notesModelChanged) {
      const newModelId = resolve_embedding_model_id(runtime.settings);
      const oldModelId = resolve_embedding_model_id(previousSettings);

      if (runtime.settings.experimental_user_data_index && newModelId) {
        let coverageStats: ModelCoverageStats | null = null;
        try {
          coverageStats = await estimate_model_coverage(newModelId);
        } catch (error) {
          runtime.log.warn('Jarvis: failed to estimate model coverage', { modelId: newModelId, error });
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

      if (runtime.settings.experimental_user_data_index) {
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
          const updateOptions = forceUpdateAfterReload
            ? { force: true }
            : { force: false, silent: true };
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
            await updates.start_update({ force: false, silent: true });
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
 
