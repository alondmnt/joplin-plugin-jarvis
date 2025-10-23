/**
 * Cross-platform MD5 hashing utility for desktop and mobile environments.
 * 
 * Uses the 'md5' package which works on both desktop and mobile, and produces
 * identical results to Node.js crypto module. Avoids using Buffer to ensure
 * mobile compatibility.
 * 
 * The md5 package natively accepts strings, arrays, and Uint8Array, so we
 * collect all chunks and concatenate them before hashing.
 * 
 * This ensures that hashes are consistent across platforms, which is critical
 * for synced databases where change detection relies on content hashes.
 */

const md5Fn = require('md5');

export interface HashInterface {
  update(data: string | Buffer | Uint8Array): HashInterface;
  digest(encoding: string): string;
}

class MD5HashWrapper implements HashInterface {
  private chunks: Array<string | Uint8Array>;

  constructor() {
    this.chunks = [];
  }

  update(data: string | Buffer | Uint8Array): HashInterface {
    if (typeof data === 'string') {
      this.chunks.push(data);
    } else {
      // Convert any binary data (Uint8Array, Buffer, etc.) to Uint8Array
      // Buffer extends Uint8Array on desktop, but we cast to avoid type issues
      this.chunks.push((data instanceof Uint8Array ? data : new Uint8Array(data as any)) as Uint8Array);
    }
    return this;
  }

  digest(encoding: string): string {
    if (encoding !== 'hex') {
      throw new Error(`Unsupported encoding: ${encoding}`);
    }
    
    // If single chunk, pass directly to md5
    if (this.chunks.length === 1) {
      return md5Fn(this.chunks[0]);
    }
    
    // Multiple chunks: need to concatenate
    // Convert everything to byte arrays and concatenate
    let totalLength = 0;
    const byteArrays: Uint8Array[] = [];
    
    for (const chunk of this.chunks) {
      if (typeof chunk === 'string') {
        const encoder = new TextEncoder();
        const bytes = encoder.encode(chunk);
        byteArrays.push(bytes);
        totalLength += bytes.length;
      } else {
        byteArrays.push(chunk);
        totalLength += chunk.length;
      }
    }
    
    // Concatenate all byte arrays
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const arr of byteArrays) {
      combined.set(arr, offset);
      offset += arr.length;
    }
    
    return md5Fn(combined);
  }
}

/**
 * Create a hash object that works across desktop and mobile platforms.
 * Produces identical MD5 hashes on both platforms for the same input.
 * 
 * @param algorithm - Hash algorithm (only 'md5' is supported)
 * @returns Hash object with update() and digest() methods
 */
export function createHash(algorithm: 'md5'): HashInterface {
  if (algorithm !== 'md5') {
    throw new Error(`Unsupported hash algorithm: ${algorithm}. Only 'md5' is supported.`);
  }
  return new MD5HashWrapper();
}
