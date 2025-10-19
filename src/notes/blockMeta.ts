import { BlockEmbedding } from './embeddings';
import { BlockRowMeta } from './userDataStore';

export interface BlockMetaOptions {
  headingPaths?: string[][];
  tagsPerBlock?: string[][];
  blockIdPrefix?: string;
}

/**
 * Convert runtime `BlockEmbedding` objects into the metadata rows stored alongside
 * q8 shards. Optional heading paths / tags can be supplied for richer context.
 */
export function buildBlockRowMeta(blocks: BlockEmbedding[], options: BlockMetaOptions = {}): BlockRowMeta[] {
  const { headingPaths = [], tagsPerBlock = [], blockIdPrefix } = options;
  return blocks.map((block, index) => {
    const headingPath = headingPaths[index] ?? (block.title ? [block.title] : []);
    const tags = tagsPerBlock[index];
    const blockId = `${blockIdPrefix ?? block.id}:${index}:v1`; // keep deterministic across rebuilds
    return {
      blockId,
      noteId: block.id,
      noteHash: block.hash,
      title: block.title,
      headingLevel: block.level,
      headingPath,
      bodyStart: block.body_idx,
      bodyLength: block.length,
      lineNumber: block.line,
      tags: tags && tags.length > 0 ? tags : undefined,
    };
  });
}
