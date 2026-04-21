#pragma once
/**
 * ============================================================
 *  huffman.h  —  Canonical Huffman Coder
 *  CompressedLLM / General purpose text compression
 * ============================================================
 *
 *  WHY CANONICAL HUFFMAN:
 *    Standard Huffman stores the full tree structure in the header.
 *    Canonical Huffman stores ONLY code lengths (one byte per symbol).
 *    The decoder rebuilds the tree deterministically from lengths alone.
 *
 *    Header cost:
 *      Standard:   O(symbols × log(symbols)) bits
 *      Canonical:  exactly 256 bytes (one length byte per symbol) — FIXED
 *
 *    This makes the header tiny and constant-size regardless of input.
 *
 *  BITSTREAM:
 *    Bits are packed MSB-first into bytes.
 *    A 1-byte trailer stores the number of padding bits in the last byte.
 *
 *  SYMBOLS:
 *    We encode 256 possible byte values (0x00–0xFF).
 *    Only symbols that actually appear get codes.
 *    Absent symbols get length=0 (no code assigned).
 * ============================================================
 */

#ifndef CLM_HUFFMAN_H
#define CLM_HUFFMAN_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <stdexcept>

namespace clm {

// ─────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────

constexpr int HUFF_SYMBOLS   = 256;   // byte alphabet
constexpr int HUFF_MAX_BITS  = 15;    // max code length (fits in uint16_t)

// ─────────────────────────────────────────────
//  CODE TABLE ENTRY
// ─────────────────────────────────────────────

struct HuffCode {
    uint16_t code;    // the binary code (MSB-aligned in lower bits)
    uint8_t  length;  // number of bits (0 = symbol not present)
};

// ─────────────────────────────────────────────
//  HUFFMAN CODER CLASS
// ─────────────────────────────────────────────

class HuffmanCoder {
public:
    HuffmanCoder() = default;

    /**
     * Build Huffman codes from a frequency table.
     * @param freq  Array of 256 frequency counts (one per byte value)
     */
    void build(const uint64_t freq[HUFF_SYMBOLS]);

    /**
     * Encode a byte buffer using the current code table.
     * @param data    Input bytes
     * @param size    Input size
     * @return        Packed bitstream bytes
     */
    std::vector<uint8_t> encode(const uint8_t* data, size_t size) const;

    /**
     * Decode a packed bitstream back to bytes.
     * @param bits        Encoded bitstream
     * @param bits_size   Bitstream byte count
     * @param padding     Number of padding bits in last byte
     * @param out_size    Expected output size
     * @return            Decoded bytes
     */
    std::vector<uint8_t> decode(const uint8_t* bits, size_t bits_size,
                                uint8_t padding, size_t out_size) const;

    /**
     * Serialize code lengths to 256-byte array (canonical header).
     */
    void serialize_lengths(uint8_t out[HUFF_SYMBOLS]) const;

    /**
     * Rebuild code table from serialized lengths.
     */
    void deserialize_lengths(const uint8_t lengths[HUFF_SYMBOLS]);

    /**
     * Returns number of padding bits used in last encode() call.
     */
    uint8_t last_padding() const { return last_padding_; }

private:
    HuffCode  codes_[HUFF_SYMBOLS];   // encode table
    mutable uint8_t   last_padding_ = 0;

    // Decode table: for each possible prefix, store (symbol, length)
    // We use a flat 2^HUFF_MAX_BITS table for O(1) decode per symbol
    struct DecodeEntry {
        uint8_t symbol;
        uint8_t length;   // 0 = invalid
    };
    std::vector<DecodeEntry> decode_table_;

    void build_canonical_codes(uint8_t lengths[HUFF_SYMBOLS]);
    void build_decode_table();
};

} // namespace clm

#endif // CLM_HUFFMAN_H
