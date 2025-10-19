import joplin from 'api';
import { ModelType } from 'api/types';
import { Buffer } from 'buffer';
import { getLogger } from '../utils/logger';

const log = getLogger();

export interface AnchorMetadata {
  modelId: string;
  dim: number;
  nlist?: number;
  version?: string;
  hash?: string;
  format?: string;
  updatedAt?: string;
}

export interface CentroidPayload {
  format: 'f32' | 'f16';
  dim: number;
  nlist: number;
  version: number | string;
  b64: string;
  updatedAt?: string;
  trainedOn?: Record<string, unknown>;
  hash?: string;
}

const METADATA_KEY = 'jarvis/v1/aux/metadata';
const CENTROIDS_KEY = 'jarvis/v1/aux/centroids';
const PARENT_MAP_PREFIX = 'jarvis/v1/aux/parentMap/';

/**
 * Persist lightweight metadata about the model on its anchor note. Devices use
 * this to confirm anchor validity before consuming centroids.
 */
export async function writeAnchorMetadata(noteId: string, metadata: AnchorMetadata): Promise<void> {
  await joplin.data.userDataSet(ModelType.Note, noteId, METADATA_KEY, metadata);
  log.info('Anchor metadata updated', { noteId, modelId: metadata.modelId, version: metadata.version });
}

export async function readAnchorMetadata(noteId: string): Promise<AnchorMetadata | null> {
  try {
    const metadata = await joplin.data.userDataGet<AnchorMetadata>(ModelType.Note, noteId, METADATA_KEY);
    return metadata ?? null;
  } catch (error) {
    log.warn('Failed to read anchor metadata', { noteId, error });
    return null;
  }
}

/**
 * Store IVF centroids (base64 payload) on the anchor note.
 */
export async function writeCentroids(noteId: string, payload: CentroidPayload): Promise<void> {
  await joplin.data.userDataSet(ModelType.Note, noteId, CENTROIDS_KEY, payload);
  log.info('Anchor centroids updated', { noteId, format: payload.format, dim: payload.dim, nlist: payload.nlist });
}

export async function readCentroids(noteId: string): Promise<CentroidPayload | null> {
  try {
    const payload = await joplin.data.userDataGet<CentroidPayload>(ModelType.Note, noteId, CENTROIDS_KEY);
    return payload ?? null;
  } catch (error) {
    log.warn('Failed to read centroids', { noteId, error });
    return null;
  }
}

export async function writeParentMap(noteId: string, size: number, data: Uint16Array): Promise<void> {
  const key = `${PARENT_MAP_PREFIX}${size}`;
  const b64 = Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString('base64');
  await joplin.data.userDataSet(ModelType.Note, noteId, key, { b64, size });
  log.debug('Anchor parent map updated', { noteId, size });
}

export async function readParentMap(noteId: string, size: number): Promise<Uint16Array | null> {
  const key = `${PARENT_MAP_PREFIX}${size}`;
  try {
    const payload = await joplin.data.userDataGet<{ b64: string; size: number }>(ModelType.Note, noteId, key);
    if (!payload?.b64) {
      return null;
    }
    const buf = Buffer.from(payload.b64, 'base64');
    return new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / Uint16Array.BYTES_PER_ELEMENT);
  } catch (error) {
    log.warn('Failed to read parent map', { noteId, size, error });
    return null;
  }
}
