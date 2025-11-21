import joplin from 'api';
import { ModelType } from 'api/types';
import { getLogger } from '../utils/logger';
import { base64ToUint8Array, typedArrayToBase64 } from '../utils/base64';
import { EmbeddingSettings } from './userDataStore';

export type CentroidRefreshReason =
  | 'bootstrap'
  | 'missingPayload'
  | 'dimMismatch'
  | 'nlistMismatch'
  | 'versionMismatch'
  | 'settingsChanged'
  | 'rowGrowth'
  | 'rowShrink';

export interface AnchorRefreshState {
  status: 'pending' | 'in_progress';
  reason: CentroidRefreshReason;
  requestedAt: string;
  requestedBy?: string;
  lastAttemptAt?: string;
}

const log = getLogger();

export interface AnchorMetadata {
  modelId: string;
  dim: number;
  nlist?: number;
  version?: string;
  settings?: EmbeddingSettings; // Replaced hash with explicit settings
  format?: string;
  updatedAt?: string;
  rowCount?: number;
  centroidUpdatedAt?: string;
  centroidHash?: string;
  refresh?: AnchorRefreshState;
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

/**
 * Store IVF centroids (base64 payload) on the anchor note.
 * Validates payload size and verifies write by reading back.
 */
export async function write_centroids(noteId: string, payload: CentroidPayload): Promise<void> {
  const MIN_VALID_CLUSTERS = 2;
  
  // Validate payload before write
  if (payload.nlist < MIN_VALID_CLUSTERS) {
    throw new Error(`Invalid centroid count: ${payload.nlist} < ${MIN_VALID_CLUSTERS}`);
  }
  
  if (!payload.b64 || payload.b64.length === 0) {
    throw new Error('Centroid payload has empty base64 data');
  }
  
  // Validate base64 size matches expected centroid data size
  const expectedSize = payload.dim * payload.nlist * Float32Array.BYTES_PER_ELEMENT;
  const b64 = payload.b64;
  // Calculate actual decoded size accounting for base64 padding
  const actualSize = Math.floor((b64.length * 3) / 4) - (b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0);
  
  if (Math.abs(expectedSize - actualSize) > 16) {  // Allow small padding tolerance
    throw new Error(
      `Centroid size mismatch: expected ${expectedSize} bytes (${payload.nlist} × ${payload.dim} × 4), ` +
      `but base64 decodes to ${actualSize} bytes`
    );
  }
  
  log.info('Writing centroids', { noteId, nlist: payload.nlist, dim: payload.dim });
  
  await joplin.data.userDataSet(ModelType.Note, noteId, CENTROIDS_KEY, payload);
  
  // VERIFY write succeeded by reading back
  const readback = await read_centroids(noteId);
  if (!readback || readback.nlist !== payload.nlist || readback.dim !== payload.dim) {
    throw new Error(
      `Centroid write verification failed: wrote ${payload.nlist} centroids (dim ${payload.dim}), ` +
      `but read back ${readback?.nlist ?? 0} centroids (dim ${readback?.dim ?? 0})`
    );
  }
  
  if (!readback.b64 || Math.abs(readback.b64.length - b64.length) > 4) {
    throw new Error(
      `Centroid data size mismatch after write: wrote ${b64.length} bytes, ` +
      `read ${readback.b64?.length ?? 0} bytes`
    );
  }
  
  log.info('Centroids verified', { noteId, nlist: payload.nlist });
}

export async function read_centroids(noteId: string): Promise<CentroidPayload | null> {
  try {
    const payload = await joplin.data.userDataGet<CentroidPayload>(ModelType.Note, noteId, CENTROIDS_KEY);
    return payload ?? null;
  } catch (error) {
    log.warn('Failed to read centroids', { noteId, error });
    return null;
  }
}

export async function write_parent_map(noteId: string, size: number, data: Uint16Array): Promise<void> {
  const key = `${PARENT_MAP_PREFIX}${size}`;
  const b64 = typedArrayToBase64(data as any);
  await joplin.data.userDataSet(ModelType.Note, noteId, key, { b64, size });
  log.debug('Anchor parent map updated', { noteId, size });
}

export async function read_parent_map(noteId: string, size: number): Promise<Uint16Array | null> {
  const key = `${PARENT_MAP_PREFIX}${size}`;
  try {
    const payload = await joplin.data.userDataGet<{ b64: string; size: number }>(ModelType.Note, noteId, key);
    if (!payload?.b64) {
      return null;
    }
    const buf = base64ToUint8Array(payload.b64);
    return new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / Uint16Array.BYTES_PER_ELEMENT);
  } catch (error) {
    log.warn('Failed to read parent map', { noteId, size, error });
    return null;
  }
}
