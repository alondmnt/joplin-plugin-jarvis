/**
 * Block navigation utilities for finding related blocks within notes.
 *
 * This module provides functions to navigate between blocks in embeddings,
 * useful for context expansion and block traversal.
 */

import type { BlockEmbedding } from './embeddings';

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

