import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { EmbeddingSettings } from './userDataStore';

const log = getLogger();

/**
 * Catalog-level metadata for an embedding model.
 * Stored on the catalog note under key `jarvis/v1/models/{modelId}/metadata`.
 */
export interface CatalogModelMetadata {
  modelId: string;
  dim: number;
  version?: string;
  settings?: EmbeddingSettings;
  updatedAt?: string;
  rowCount?: number;
}

/**
 * Build the userData key for a model's metadata on the catalog note.
 */
function metadata_key(modelId: string): string {
  return `jarvis/v1/models/${modelId}/metadata`;
}

/**
 * Persist lightweight metadata about the model on the catalog note.
 *
 * @param catalogNoteId - The catalog note ID where metadata is stored
 * @param modelId - The model identifier
 * @param metadata - The metadata to persist
 */
export async function write_model_metadata(
  catalogNoteId: string,
  modelId: string,
  metadata: CatalogModelMetadata
): Promise<void> {
  await joplin.data.userDataSet(ModelType.Note, catalogNoteId, metadata_key(modelId), metadata);
  log.info('Model metadata updated', { catalogNoteId, modelId, version: metadata.version });
}

/**
 * Read metadata for a model from the catalog note.
 *
 * @param catalogNoteId - The catalog note ID where metadata is stored
 * @param modelId - The model identifier
 * @returns The metadata or null if not found
 */
export async function read_model_metadata(
  catalogNoteId: string,
  modelId: string
): Promise<CatalogModelMetadata | null> {
  try {
    const metadata = await joplin.data.userDataGet<CatalogModelMetadata>(
      ModelType.Note,
      catalogNoteId,
      metadata_key(modelId)
    );
    return metadata ?? null;
  } catch (error) {
    log.warn('Failed to read model metadata', { catalogNoteId, modelId, error });
    return null;
  }
}

/**
 * Delete metadata for a model from the catalog note.
 *
 * @param catalogNoteId - The catalog note ID where metadata is stored
 * @param modelId - The model identifier
 */
export async function delete_model_metadata(
  catalogNoteId: string,
  modelId: string
): Promise<void> {
  try {
    await joplin.data.userDataDelete(ModelType.Note, catalogNoteId, metadata_key(modelId));
    log.info('Model metadata deleted', { catalogNoteId, modelId });
  } catch (error) {
    log.warn('Failed to delete model metadata', { catalogNoteId, modelId, error });
  }
}
