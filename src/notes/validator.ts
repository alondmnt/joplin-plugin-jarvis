/**
 * Validation logic for userData embeddings during search.
 * Implements lazy validation approach: single validation point during search when embeddings are loaded.
 * Tracks mismatches and supports dialog display with human-readable diffs.
 */

import { TextEmbeddingModel } from '../models/models';
import { JarvisSettings } from '../ux/settings';
import { NoteEmbMeta, EmbeddingSettings, ModelMetadata } from './userDataStore';
import { getLogger } from '../utils/logger';

const log = getLogger();

export interface ValidationMismatch {
  noteId: string;
  mismatchType: 'model' | 'embeddingVersion' | 'modelVersion' | 'settings';
  expected: string;
  actual: string;
  details?: { [key: string]: { expected: any; actual: any } };
}

export interface ValidationResult {
  isValid: boolean;
  mismatches: ValidationMismatch[];
}

export interface ValidationStats {
  checkedCount: number;
  mismatchCount: number;
  durationMs: number;
}

/**
 * Extract embedding settings from Jarvis settings for comparison
 */
export function extract_embedding_settings_for_validation(settings: JarvisSettings): EmbeddingSettings {
  return {
    embedTitle: settings.notes_embed_title,
    embedPath: settings.notes_embed_path,
    embedHeading: settings.notes_embed_heading,
    embedTags: settings.notes_embed_tags,
    includeCode: settings.notes_include_code,
    minLength: settings.notes_min_length,
    maxTokens: settings.notes_max_tokens,
  };
}

/**
 * Compare two embedding settings objects field by field
 * Returns true if all fields match, false otherwise
 */
export function settings_equal(a: EmbeddingSettings, b: EmbeddingSettings): boolean {
  return (
    a.embedTitle === b.embedTitle &&
    a.embedPath === b.embedPath &&
    a.embedHeading === b.embedHeading &&
    a.embedTags === b.embedTags &&
    a.includeCode === b.includeCode &&
    a.minLength === b.minLength &&
    a.maxTokens === b.maxTokens
  );
}

/**
 * Get human-readable diff of two settings objects
 * Returns object with changed fields only
 */
function get_settings_diff(expected: EmbeddingSettings, actual: EmbeddingSettings): { [key: string]: { expected: any; actual: any } } {
  const diff: { [key: string]: { expected: any; actual: any } } = {};
  
  if (expected.embedTitle !== actual.embedTitle) {
    diff.embedTitle = { expected: expected.embedTitle, actual: actual.embedTitle };
  }
  if (expected.embedPath !== actual.embedPath) {
    diff.embedPath = { expected: expected.embedPath, actual: actual.embedPath };
  }
  if (expected.embedHeading !== actual.embedHeading) {
    diff.embedHeading = { expected: expected.embedHeading, actual: actual.embedHeading };
  }
  if (expected.embedTags !== actual.embedTags) {
    diff.embedTags = { expected: expected.embedTags, actual: actual.embedTags };
  }
  if (expected.includeCode !== actual.includeCode) {
    diff.includeCode = { expected: expected.includeCode, actual: actual.includeCode };
  }
  if (expected.minLength !== actual.minLength) {
    diff.minLength = { expected: expected.minLength, actual: actual.minLength };
  }
  if (expected.maxTokens !== actual.maxTokens) {
    diff.maxTokens = { expected: expected.maxTokens, actual: actual.maxTokens };
  }
  
  return diff;
}

/**
 * Validate a single note's metadata against current model and settings
 * Returns validation result with any detected mismatches
 */
export function validate_note_metadata(
  noteId: string,
  meta: NoteEmbMeta,
  currentModel: TextEmbeddingModel,
  currentSettings: EmbeddingSettings
): ValidationResult {
  const mismatches: ValidationMismatch[] = [];
  
  // Check if active model matches current model
  if (meta.activeModelId !== currentModel.id) {
    mismatches.push({
      noteId,
      mismatchType: 'model',
      expected: currentModel.id,
      actual: meta.activeModelId,
    });
  }
  
  // Get model metadata for active model
  const modelMeta = meta.models[meta.activeModelId];
  if (!modelMeta) {
    // Inconsistent metadata: has activeModelId but no model entry
    log.warn(`Note ${noteId}: inconsistent metadata (activeModelId ${meta.activeModelId} not in models), skipping validation`);
    return { isValid: false, mismatches: [] };
  }
  
  // Check embedding version
  if (modelMeta.embeddingVersion !== currentModel.embedding_version) {
    mismatches.push({
      noteId,
      mismatchType: 'embeddingVersion',
      expected: String(currentModel.embedding_version),
      actual: String(modelMeta.embeddingVersion),
    });
  }
  
  // Check model version
  if (modelMeta.modelVersion !== currentModel.version) {
    mismatches.push({
      noteId,
      mismatchType: 'modelVersion',
      expected: currentModel.version,
      actual: modelMeta.modelVersion,
    });
  }
  
  // Check settings (field-by-field comparison)
  if (!settings_equal(currentSettings, modelMeta.settings)) {
    const settingsDiff = get_settings_diff(currentSettings, modelMeta.settings);
    mismatches.push({
      noteId,
      mismatchType: 'settings',
      expected: JSON.stringify(currentSettings),
      actual: JSON.stringify(modelMeta.settings),
      details: settingsDiff,
    });
  }
  
  return {
    isValid: mismatches.length === 0,
    mismatches,
  };
}

/**
 * Session-level validator that tracks validation state across searches
 * Implements "show dialog once per session" logic for mismatches
 */
export class ValidationTracker {
  private hasShownDialogThisSession: boolean = false;
  private lastValidationStats: ValidationStats | null = null;
  private allMismatches: ValidationMismatch[] = [];
  
  /**
   * Reset validation state (e.g., for testing or manual reset)
   */
  reset(): void {
    this.hasShownDialogThisSession = false;
    this.lastValidationStats = null;
    this.allMismatches = [];
  }
  
  /**
   * Check if dialog should be shown for current mismatches
   */
  should_show_dialog(): boolean {
    return !this.hasShownDialogThisSession && this.allMismatches.length > 0;
  }
  
  /**
   * Mark dialog as shown for this session
   */
  mark_dialog_shown(): void {
    this.hasShownDialogThisSession = true;
  }
  
  /**
   * Get all accumulated mismatches
   */
  get_mismatches(): ValidationMismatch[] {
    return this.allMismatches;
  }
  
  /**
   * Get last validation stats
   */
  get_stats(): ValidationStats | null {
    return this.lastValidationStats;
  }
  
  /**
   * Validate multiple notes and accumulate mismatches
   * Returns validation stats for logging/monitoring
   */
  validate_notes(
    notesMeta: Array<{ noteId: string; meta: NoteEmbMeta }>,
    currentModel: TextEmbeddingModel,
    currentSettings: EmbeddingSettings
  ): ValidationStats {
    const startTime = Date.now();
    const newMismatches: ValidationMismatch[] = [];
    
    for (const { noteId, meta } of notesMeta) {
      const result = validate_note_metadata(noteId, meta, currentModel, currentSettings);
      newMismatches.push(...result.mismatches);
    }
    
    // Accumulate mismatches (de-duplicate by noteId + mismatchType)
    const existingKeys = new Set(
      this.allMismatches.map(m => `${m.noteId}:${m.mismatchType}`)
    );
    for (const mismatch of newMismatches) {
      const key = `${mismatch.noteId}:${mismatch.mismatchType}`;
      if (!existingKeys.has(key)) {
        this.allMismatches.push(mismatch);
        existingKeys.add(key);
      }
    }
    
    const stats: ValidationStats = {
      checkedCount: notesMeta.length,
      mismatchCount: newMismatches.length,
      durationMs: Date.now() - startTime,
    };
    
    this.lastValidationStats = stats;
    
    // Log performance metrics
    log.info(
      `Validation: checked ${stats.checkedCount} notes in ${stats.durationMs}ms, ` +
      `${stats.mismatchCount} mismatches (${this.allMismatches.length} total accumulated)`
    );
    
    return stats;
  }
  
  /**
   * Format mismatches for display in dialog
   * Returns human-readable summary grouped by mismatch type
   */
  format_mismatches_for_dialog(): string {
    if (this.allMismatches.length === 0) {
      return '';
    }
    
    // Group by mismatch type
    const byType: { [type: string]: ValidationMismatch[] } = {};
    for (const mismatch of this.allMismatches) {
      if (!byType[mismatch.mismatchType]) {
        byType[mismatch.mismatchType] = [];
      }
      byType[mismatch.mismatchType].push(mismatch);
    }
    
    const lines: string[] = [];
    
    if (byType.model) {
      lines.push(`[${byType.model.length} notes: wrong model]`);
    }
    
    if (byType.embeddingVersion) {
      lines.push(`[${byType.embeddingVersion.length} notes: older embedding version]`);
    }
    
    if (byType.modelVersion) {
      lines.push(`[${byType.modelVersion.length} notes: older model version]`);
    }
    
    if (byType.settings && byType.settings.length > 0) {
      // Get common setting changes across all mismatched notes
      const settingChanges = new Set<string>();
      for (const mismatch of byType.settings) {
        if (mismatch.details) {
          for (const key of Object.keys(mismatch.details)) {
            const { expected, actual } = mismatch.details[key];
            settingChanges.add(`${key}: ${actual}→${expected}`);
          }
        }
      }
      const changesStr = Array.from(settingChanges).join(', ');
      lines.push(`[${byType.settings.length} notes: different settings: ${changesStr}]`);
    }
    
    return lines.join(' ');
  }
}

/**
 * Global validation tracker instance (singleton for session-level tracking)
 */
export const globalValidationTracker = new ValidationTracker();

