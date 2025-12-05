/**
 * Cross-platform base64 encoding/decoding utilities.
 * 
 * On desktop, uses Buffer for efficiency.
 * On mobile, uses browser-native atob/btoa with proper binary handling.
 * 
 * These functions produce identical results across platforms.
 */

// Try to detect if Buffer is available (desktop)
let hasBuffer = false;
try {
  const BufferCheck = require('buffer').Buffer;
  hasBuffer = typeof BufferCheck !== 'undefined';
} catch (e) {
  hasBuffer = false;
}

/**
 * Convert a Uint8Array to base64 string.
 * Works on both desktop and mobile.
 */
export function uint8ArrayToBase64(data: Uint8Array): string {
  if (hasBuffer) {
    // Desktop: use Buffer for efficiency
    const Buffer = require('buffer').Buffer;
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString('base64');
  } else {
    // Mobile: use browser's btoa with proper binary conversion
    let binaryString = '';
    for (let i = 0; i < data.length; i++) {
      binaryString += String.fromCharCode(data[i]);
    }
    return btoa(binaryString);
  }
}

/**
 * Convert a base64 string to Uint8Array.
 * Works on both desktop and mobile.
 */
export function base64ToUint8Array(base64: string): Uint8Array {
  if (hasBuffer) {
    // Desktop: use Buffer for efficiency
    const Buffer = require('buffer').Buffer;
    const buf = Buffer.from(base64, 'base64');
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } else {
    // Mobile: use browser's atob
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }
}

/**
 * Helper to convert TypedArray to base64.
 * Works with Float32Array, Uint16Array, etc.
 */
export function typedArrayToBase64(data: { buffer: ArrayBuffer; byteOffset: number; byteLength: number }): string {
  const uint8View = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return uint8ArrayToBase64(uint8View);
}
