import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { EmbeddingSettings } from './userDataStore';

const log = getLogger();

export interface AnchorMetadata {
  modelId: string;
  dim: number;
  version?: string;
  settings?: EmbeddingSettings;
  updatedAt?: string;
  rowCount?: number;
}

const METADATA_KEY = 'jarvis/v1/aux/metadata';

/**
 * Persist lightweight metadata about the model on its anchor note.
 */
export async function write_anchor_metadata(noteId: string, metadata: AnchorMetadata): Promise<void> {
  await joplin.data.userDataSet(ModelType.Note, noteId, METADATA_KEY, metadata);
  log.info('Anchor metadata updated', { noteId, modelId: metadata.modelId, version: metadata.version });
}

export async function read_anchor_meta_data(noteId: string): Promise<AnchorMetadata | null> {
  try {
    const metadata = await joplin.data.userDataGet<AnchorMetadata>(ModelType.Note, noteId, METADATA_KEY);
    return metadata ?? null;
  } catch (error) {
    log.warn('Failed to read anchor metadata', { noteId, error });
    return null;
  }
}
