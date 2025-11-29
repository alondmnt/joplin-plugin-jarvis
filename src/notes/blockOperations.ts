/**
 * Block navigation utilities for finding related blocks within notes.
 *
 * This module provides functions to navigate between blocks in embeddings,
 * useful for context expansion and block traversal.
 */

import type { BlockEmbedding } from './embeddings';
import { JarvisSettings } from '../ux/settings';

/**
 * Calculate cosine similarity between two Float32 embeddings.
 * Imported from embeddings.ts to avoid duplication.
 */
import { calc_similarity } from './embeddings';

/**
 * Find the next n blocks in the same note after the given block.
 * Blocks are ordered by line number.
 *
 * @param block - The reference block
 * @param embeddings - All available embeddings to search through
 * @param n - Number of blocks to return (default: 1)
 * @returns Array of next blocks, or empty array if none found
 */
export async function get_next_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], n: number = 1): Promise<BlockEmbedding[]> {
  const next_blocks = embeddings.filter((embd) => embd.id === block.id && embd.line > block.line)
    .sort((a, b) => a.line - b.line);
  if (next_blocks.length === 0) {
    return [];
  }
  return next_blocks.slice(0, n);
}

/**
 * Find the previous n blocks in the same note before the given block.
 * Blocks are ordered by line number (descending).
 *
 * @param block - The reference block
 * @param embeddings - All available embeddings to search through
 * @param n - Number of blocks to return (default: 1)
 * @returns Array of previous blocks, or empty array if none found
 */
export async function get_prev_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], n: number = 1): Promise<BlockEmbedding[]> {
  const prev_blocks = embeddings.filter((embd) => embd.id === block.id && embd.line < block.line)
    .sort((a, b) => b.line - a.line);
  if (prev_blocks.length === 0) {
    return [];
  }
  return prev_blocks.slice(0, n);
}

/**
 * Find the n most semantically similar blocks to the given block.
 * Uses cosine similarity between embeddings and filters by similarity threshold.
 *
 * @param block - The reference block to find similar blocks for
 * @param embeddings - All available embeddings to search through
 * @param settings - Jarvis settings (for min_similarity and min_length thresholds)
 * @param n - Number of similar blocks to return (default: 1)
 * @returns Array of nearest blocks sorted by similarity (excluding the input block itself)
 */
export async function get_nearest_blocks(block: BlockEmbedding, embeddings: BlockEmbedding[], settings: JarvisSettings, n: number = 1): Promise<BlockEmbedding[]> {
  // Calculate similarity for each embedding
  // see also find_nearest_notes
  const nearest = embeddings.map(
    (embd: BlockEmbedding): BlockEmbedding => {
    const new_embd = Object.assign({}, embd);
    new_embd.similarity = calc_similarity(block.embedding, new_embd.embedding);
    return new_embd;
  }
  // Filter by similarity and length thresholds
  ).filter((embd) => (embd.similarity >= settings.notes_min_similarity) && (embd.length >= settings.notes_min_length));

  // Sort by similarity (highest first) and skip the first item (the block itself)
  return nearest.sort((a, b) => b.similarity - a.similarity).slice(1, n+1);
}
