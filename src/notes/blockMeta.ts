import { BlockEmbedding } from './embeddings';
import { BlockRowMeta } from './userDataStore';

export interface BlockMetaOptions {
  headingPaths?: string[][];
  tagsPerBlock?: string[][];
}

/**
 * Convert runtime `BlockEmbedding` objects into the metadata rows stored alongside
 * q8 shards. Optional heading paths / tags can be supplied for richer context.
 */
export function build_block_row_meta(blocks: BlockEmbedding[], options: BlockMetaOptions = {}): BlockRowMeta[] {
  const { headingPaths = [], tagsPerBlock = [] } = options;
  return blocks.map((block, index) => {
    const headingPath = headingPaths[index] ?? (block.title ? [block.title] : []);
    const tags = tagsPerBlock[index];
    const meta: BlockRowMeta = {
      title: block.title,
      headingLevel: block.level,
      bodyStart: block.body_idx,
      bodyLength: block.length,
      lineNumber: block.line,
      tags: tags && tags.length > 0 ? tags : undefined,
    };
    // Avoid storing headingPath when it only duplicates the title
    if (headingPath && !(headingPath.length === 1 && headingPath[0] === block.title)) {
      meta.headingPath = headingPath;
    }
    return meta;
  });
}
