/**
 * Fixed-capacity min-heap that keeps the top scoring entries. The smallest score
 * lives at the root so pushes discard low-quality candidates once capacity is
 * reached. Designed for the semantic search ranking path where K is small.
 */
export class TopKHeap<T> {
  private readonly capacity: number;
  private readonly minScore: number;
  private heap: Array<{ score: number; value: T }> = [];

  constructor(capacity: number, options: { minScore?: number } = {}) {
    this.capacity = Math.max(0, capacity);
    this.minScore = options.minScore ?? -Infinity;
  }

  get size(): number {
    return this.heap.length;
  }

  /**
   * Attempt to insert a candidate. Returns true when the value is accepted, false otherwise.
   */
  push(score: number, value: T): boolean {
    if (this.capacity === 0) {
      return false;
    }
    if (score < this.minScore) {
      return false;
    }
    if (this.heap.length < this.capacity) {
      this.heap.push({ score, value });
      this.siftUp(this.heap.length - 1);
      return true;
    }
    if (score <= this.heap[0].score) {
      return false;
    }
    this.heap[0] = { score, value };
    this.siftDown(0);
    return true;
  }

  /**
   * Return heap contents ordered from highest to lowest score.
   */
  valuesDescending(): Array<{ score: number; value: T }> {
    return [...this.heap].sort((a, b) => b.score - a.score);
  }

  /**
   * Bubble an entry upward until the min-heap property holds.
   */
  private siftUp(index: number): void {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.heap[parent].score <= this.heap[index].score) {
        break;
      }
      this.swap(index, parent);
      index = parent;
    }
  }

  /**
   * Push an entry down the tree until the min-heap property holds.
   */
  private siftDown(index: number): void {
    const length = this.heap.length;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;

      if (left < length && this.heap[left].score < this.heap[smallest].score) {
        smallest = left;
      }
      if (right < length && this.heap[right].score < this.heap[smallest].score) {
        smallest = right;
      }
      if (smallest === index) {
        break;
      }
      this.swap(index, smallest);
      index = smallest;
    }
  }

  /**
   * Swap two positions inside the backing array.
   */
  private swap(a: number, b: number): void {
    const tmp = this.heap[a];
    this.heap[a] = this.heap[b];
    this.heap[b] = tmp;
  }
}
