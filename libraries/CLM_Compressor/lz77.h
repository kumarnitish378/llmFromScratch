#pragma once
/**
 * ============================================================
 *  lz77.h  —  Hash-Chain LZ77 Compressor
 *  CompressedLLM / General purpose text compression
 * ============================================================
 *
 *  ALGORITHM: Hash-chain sliding window (same core as zlib/gzip)
 *
 *  WHY THIS IS FAST (vs Python O(n²)):
 *    Python version: for every input byte, scan up to 255 positions
 *                    = O(n × window) pure Python bytecode loops
 *
 *    This version:   hash the next 3 bytes → jump directly to all
 *                    previous positions with same hash → O(n) average
 *                    with O(chain_depth) worst case (bounded, default=64)
 *
 *  LAZY MATCHING:
 *    Instead of greedily taking the first good match, we look one
 *    byte ahead. If position i+1 gives a longer match than position i,
 *    emit a literal at i and use the better match at i+1.
 *    This improves compression ratio by ~5-10% at small speed cost.
 *
 *  TOKEN ENCODING (compact binary, not Python's char-based):
 *    Literal:    [0] [byte]           → 2 bytes
 *    Reference:  [1] [offset_lo]      → 4 bytes
 *                    [offset_hi]
 *                    [length]
 *    offset: 16-bit (up to 65535 bytes lookback window)
 *    length: 8-bit  (3–255 match length)
 *
 *  MEMORY:
 *    Hash table:  65536 × 4 bytes = 256KB  (fixed)
 *    Chain table: window_size × 4 bytes = 256KB (default 64K window)
 *    Input buffer: streaming, not copied
 *    Token buffer: 3-4 bytes/token vs Python's 200 bytes/token
 * ============================================================
 */

#ifndef CLM_LZ77_H
#define CLM_LZ77_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace clm {

// ─────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────

constexpr uint16_t LZ77_WINDOW_BITS  = 16;
constexpr uint32_t LZ77_WINDOW_SIZE  = 1u << LZ77_WINDOW_BITS;   // 65536
constexpr uint32_t LZ77_WINDOW_MASK  = LZ77_WINDOW_SIZE - 1;

constexpr uint16_t LZ77_HASH_BITS    = 16;
constexpr uint32_t LZ77_HASH_SIZE    = 1u << LZ77_HASH_BITS;     // 65536 buckets
constexpr uint32_t LZ77_HASH_MASK    = LZ77_HASH_SIZE - 1;

constexpr uint8_t  LZ77_MIN_MATCH    = 3;
constexpr uint8_t  LZ77_MAX_MATCH    = 255;
constexpr uint32_t LZ77_MAX_CHAIN    = 64;    // max chain depth per lookup (speed/ratio tradeoff)
constexpr uint32_t LZ77_LAZY_MATCH   = 8;     // min length gain to trigger lazy match

// Token type flags in serialized stream
constexpr uint8_t  LZ77_FLAG_LITERAL = 0x00;
constexpr uint8_t  LZ77_FLAG_REF     = 0x01;

// ─────────────────────────────────────────────
//  TOKEN (in-memory representation)
// ─────────────────────────────────────────────

struct LZ77Token {
    uint8_t  flag;       // LZ77_FLAG_LITERAL or LZ77_FLAG_REF
    uint8_t  byte_val;   // literal byte (flag == LITERAL)
    uint16_t offset;     // back-reference offset (flag == REF)
    uint8_t  length;     // back-reference length (flag == REF)
};

// ─────────────────────────────────────────────
//  COMPRESSOR CLASS
// ─────────────────────────────────────────────

class LZ77Compressor {
public:
    LZ77Compressor();

    /**
     * Compress input bytes into LZ77 tokens.
     * @param data     Pointer to input buffer
     * @param size     Size of input in bytes
     * @param lazy     Enable lazy matching (better ratio, slightly slower)
     * @return         Vector of LZ77 tokens
     */
    std::vector<LZ77Token> compress(const uint8_t* data, size_t size, bool lazy = true);

    /**
     * Decompress LZ77 tokens back to original bytes.
     * @param tokens   Tokens from compress()
     * @param orig_size Expected output size (for pre-allocation)
     * @return         Reconstructed bytes
     */
    std::vector<uint8_t> decompress(const std::vector<LZ77Token>& tokens, size_t orig_size = 0);

    /**
     * Serialize tokens to a compact binary byte stream.
     * Format per token:
     *   Literal: [0x00][byte]
     *   Ref:     [0x01][offset_lo][offset_hi][length]
     */
    static std::vector<uint8_t> serialize(const std::vector<LZ77Token>& tokens);

    /**
     * Deserialize binary stream back to tokens.
     */
    static std::vector<LZ77Token> deserialize(const uint8_t* data, size_t size);

private:
    // Hash table: maps 3-byte hash → most recent position with that hash
    uint32_t head_[LZ77_HASH_SIZE];

    // Chain table: prev_[pos & WINDOW_MASK] = previous pos with same hash
    uint32_t prev_[LZ77_WINDOW_SIZE];

    // Inline 3-byte hash (fast, minimal collisions for text)
    static inline uint32_t hash3(const uint8_t* p) {
        return ((uint32_t)p[0] * 2654435761u ^
                (uint32_t)p[1] * 40503u ^
                (uint32_t)p[2]) & LZ77_HASH_MASK;
    }

    // Insert position into hash chain, return previous head
    inline uint32_t insert_hash(uint32_t pos, const uint8_t* data) {
        uint32_t h = hash3(data + pos);
        uint32_t prev = head_[h];
        head_[h] = pos;
        prev_[pos & LZ77_WINDOW_MASK] = prev;
        return prev;
    }

    // Find longest match starting at `pos` using hash chain
    uint32_t find_match(const uint8_t* data, size_t size,
                        uint32_t pos, uint32_t& match_offset);
};

} // namespace clm

#endif // CLM_LZ77_H
