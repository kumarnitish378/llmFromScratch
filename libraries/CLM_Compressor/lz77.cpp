/**
 * lz77.cpp — Hash-Chain LZ77 Compressor Implementation
 */

#include "lz77.h"
#include <cstring>
#include <algorithm>

namespace clm {

// ─────────────────────────────────────────────
//  CONSTRUCTOR
// ─────────────────────────────────────────────

LZ77Compressor::LZ77Compressor() {
    std::memset(head_, 0xFF, sizeof(head_));   // 0xFFFFFFFF = "no entry"
    std::memset(prev_, 0xFF, sizeof(prev_));
}

// ─────────────────────────────────────────────
//  FIND MATCH  (hash chain walk)
// ─────────────────────────────────────────────

uint32_t LZ77Compressor::find_match(const uint8_t* data, size_t size,
                                     uint32_t pos, uint32_t& match_offset)
{
    if (pos + LZ77_MIN_MATCH >= size) return 0;

    uint32_t best_len = LZ77_MIN_MATCH - 1;
    uint32_t best_pos = 0;

    // Hash the 3-byte sequence at current position
    uint32_t h = hash3(data + pos);
    uint32_t candidate = head_[h];

    uint32_t chain_depth = 0;
    const uint8_t* cur = data + pos;

    while (candidate != 0xFFFFFFFF && chain_depth < LZ77_MAX_CHAIN) {
        // Candidate must be within window
        if (candidate >= pos) break;
        if (pos - candidate > LZ77_WINDOW_SIZE) break;

        const uint8_t* ref = data + candidate;
        uint32_t max_len = std::min((uint32_t)(size - pos), (uint32_t)LZ77_MAX_MATCH);

        // Fast first-byte check before full comparison
        if (ref[0] == cur[0] && ref[best_len] == cur[best_len]) {
            uint32_t len = 0;
            while (len < max_len && ref[len] == cur[len])
                ++len;

            if (len > best_len) {
                best_len = len;
                best_pos = candidate;
                if (best_len == LZ77_MAX_MATCH) break;  // can't do better
            }
        }

        candidate = prev_[candidate & LZ77_WINDOW_MASK];
        ++chain_depth;
    }

    if (best_len >= LZ77_MIN_MATCH) {
        match_offset = pos - best_pos;
        return best_len;
    }
    return 0;
}

// ─────────────────────────────────────────────
//  COMPRESS
// ─────────────────────────────────────────────

std::vector<LZ77Token> LZ77Compressor::compress(const uint8_t* data, size_t size, bool lazy)
{
    // Reset hash tables for fresh compression
    std::memset(head_, 0xFF, sizeof(head_));
    std::memset(prev_, 0xFF, sizeof(prev_));

    std::vector<LZ77Token> tokens;
    tokens.reserve(size / 2);  // pre-allocate conservatively

    uint32_t i = 0;

    while (i < size) {
        // Need at least MIN_MATCH bytes for a reference
        if (i + LZ77_MIN_MATCH > size) {
            // Emit remaining bytes as literals
            while (i < size) {
                tokens.push_back({LZ77_FLAG_LITERAL, data[i], 0, 0});
                ++i;
            }
            break;
        }

        uint32_t offset1 = 0;
        uint32_t len1 = find_match(data, size, i, offset1);

        // Insert current position into hash chain
        insert_hash(i, data);

        if (len1 < LZ77_MIN_MATCH) {
            // No match — emit literal
            tokens.push_back({LZ77_FLAG_LITERAL, data[i], 0, 0});
            ++i;
            continue;
        }

        if (lazy && i + 1 + LZ77_MIN_MATCH <= size) {
            // ── LAZY MATCHING ────────────────────────
            // Look one position ahead — maybe a longer match exists at i+1
            uint32_t offset2 = 0;
            insert_hash(i + 1, data);
            uint32_t len2 = find_match(data, size, i + 1, offset2);

            if (len2 > len1 + LZ77_LAZY_MATCH) {
                // The match at i+1 is meaningfully longer
                // Emit literal at i, use better match at i+1
                tokens.push_back({LZ77_FLAG_LITERAL, data[i], 0, 0});
                ++i;

                // Insert i+1 into hash and use its match
                insert_hash(i, data);

                if (len2 >= LZ77_MIN_MATCH) {
                    tokens.push_back({LZ77_FLAG_REF, 0,
                                     (uint16_t)offset2,
                                     (uint8_t)std::min(len2, (uint32_t)LZ77_MAX_MATCH)});
                    // Insert intermediate positions into hash chain
                    uint32_t match_end = i + len2;
                    for (uint32_t k = i + 1; k < match_end && k + LZ77_MIN_MATCH <= size; ++k)
                        insert_hash(k, data);
                    i = match_end;
                    continue;
                }
            }
        }

        // Use match at i
        uint32_t use_len = std::min(len1, (uint32_t)LZ77_MAX_MATCH);
        tokens.push_back({LZ77_FLAG_REF, 0, (uint16_t)offset1, (uint8_t)use_len});

        // Insert intermediate positions into hash chain
        uint32_t match_end = i + use_len;
        for (uint32_t k = i + 1; k < match_end && k + LZ77_MIN_MATCH <= size; ++k)
            insert_hash(k, data);
        i = match_end;
    }

    return tokens;
}

// ─────────────────────────────────────────────
//  DECOMPRESS
// ─────────────────────────────────────────────

std::vector<uint8_t> LZ77Compressor::decompress(const std::vector<LZ77Token>& tokens,
                                                  size_t orig_size)
{
    std::vector<uint8_t> out;
    out.reserve(orig_size > 0 ? orig_size : tokens.size() * 2);

    for (const auto& tok : tokens) {
        if (tok.flag == LZ77_FLAG_LITERAL) {
            out.push_back(tok.byte_val);
        } else {
            // Back-reference
            if (tok.offset == 0 || tok.offset > out.size())
                throw std::runtime_error("LZ77: invalid back-reference offset");

            size_t start = out.size() - tok.offset;
            // Note: length can exceed offset (run-length encoding case)
            for (uint8_t k = 0; k < tok.length; ++k)
                out.push_back(out[start + k % tok.offset]);
        }
    }

    return out;
}

// ─────────────────────────────────────────────
//  SERIALIZE TOKENS → BINARY
// ─────────────────────────────────────────────

std::vector<uint8_t> LZ77Compressor::serialize(const std::vector<LZ77Token>& tokens)
{
    std::vector<uint8_t> out;
    // Each token is at most 4 bytes
    out.reserve(tokens.size() * 3);

    for (const auto& tok : tokens) {
        if (tok.flag == LZ77_FLAG_LITERAL) {
            out.push_back(LZ77_FLAG_LITERAL);
            out.push_back(tok.byte_val);
        } else {
            out.push_back(LZ77_FLAG_REF);
            out.push_back(tok.offset & 0xFF);          // offset low byte
            out.push_back((tok.offset >> 8) & 0xFF);   // offset high byte
            out.push_back(tok.length);
        }
    }

    return out;
}

// ─────────────────────────────────────────────
//  DESERIALIZE BINARY → TOKENS
// ─────────────────────────────────────────────

std::vector<LZ77Token> LZ77Compressor::deserialize(const uint8_t* data, size_t size)
{
    std::vector<LZ77Token> tokens;
    tokens.reserve(size / 2);

    size_t i = 0;
    while (i < size) {
        uint8_t flag = data[i++];

        if (flag == LZ77_FLAG_LITERAL) {
            if (i >= size) throw std::runtime_error("LZ77: truncated literal");
            uint8_t b = data[i++];
            tokens.push_back({LZ77_FLAG_LITERAL, b, 0, 0});
        } else if (flag == LZ77_FLAG_REF) {
            if (i + 2 >= size) throw std::runtime_error("LZ77: truncated reference");
            uint16_t off = (uint16_t)data[i] | ((uint16_t)data[i+1] << 8);
            uint8_t  len = data[i+2];
            tokens.push_back({LZ77_FLAG_REF, 0, off, len});
            i += 3;
        } else {
            throw std::runtime_error("LZ77: invalid token flag");
        }
    }

    return tokens;
}

} // namespace clm
