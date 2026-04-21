/**
 * huffman.cpp — Canonical Huffman Coder Implementation
 */

#include "huffman.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace clm {

// ─────────────────────────────────────────────
//  INTERNAL: TREE NODE (build only, not stored)
// ─────────────────────────────────────────────

struct HuffNode {
    uint64_t freq;
    int      symbol;   // -1 = internal node
    int      left, right;  // indices into node pool, -1 = none
};

// ─────────────────────────────────────────────
//  BUILD FROM FREQUENCIES
// ─────────────────────────────────────────────

void HuffmanCoder::build(const uint64_t freq[HUFF_SYMBOLS])
{
    // Step 1: create leaf nodes for all symbols with freq > 0
    std::vector<HuffNode> nodes;
    nodes.reserve(HUFF_SYMBOLS * 2);

    std::vector<int> heap;  // indices into nodes, min-heap by freq

    for (int s = 0; s < HUFF_SYMBOLS; ++s) {
        if (freq[s] > 0) {
            int idx = (int)nodes.size();
            nodes.push_back({freq[s], s, -1, -1});
            heap.push_back(idx);
        }
    }

    // Edge case: empty input or single symbol
    if (heap.empty()) {
        std::memset(codes_, 0, sizeof(codes_));
        return;
    }
    if (heap.size() == 1) {
        // Single symbol: assign code "0"
        uint8_t lengths[HUFF_SYMBOLS] = {};
        lengths[nodes[heap[0]].symbol] = 1;
        build_canonical_codes(lengths);
        return;
    }

    // Step 2: Huffman tree via min-heap
    auto cmp = [&](int a, int b){ return nodes[a].freq > nodes[b].freq; };
    std::make_heap(heap.begin(), heap.end(), cmp);

    while (heap.size() > 1) {
        // Pop two lowest-frequency nodes
        std::pop_heap(heap.begin(), heap.end(), cmp);
        int a = heap.back(); heap.pop_back();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        int b = heap.back(); heap.pop_back();

        // Merge into internal node
        int idx = (int)nodes.size();
        nodes.push_back({nodes[a].freq + nodes[b].freq, -1, a, b});
        heap.push_back(idx);
        std::push_heap(heap.begin(), heap.end(), cmp);
    }

    // Step 3: Traverse tree to get code lengths
    uint8_t lengths[HUFF_SYMBOLS] = {};
    int root = heap[0];

    // Iterative DFS using explicit stack
    struct Frame { int node; int depth; };
    std::vector<Frame> stack;
    Frame root_frame;
    root_frame.node = root;
    root_frame.depth = 0;
    stack.push_back(root_frame);

    while (!stack.empty()) {
        Frame top = stack.back();
        stack.pop_back();
        int n = top.node;
        int d = top.depth;

        if (nodes[n].symbol >= 0) {
            // Leaf node
            lengths[nodes[n].symbol] = (uint8_t)std::min(d, HUFF_MAX_BITS);
        } else {
            if (nodes[n].left >= 0) {
                Frame left_frame;
                left_frame.node = nodes[n].left;
                left_frame.depth = d + 1;
                stack.push_back(left_frame);
            }
            if (nodes[n].right >= 0) {
                Frame right_frame;
                right_frame.node = nodes[n].right;
                right_frame.depth = d + 1;
                stack.push_back(right_frame);
            }
        }
    }

    // Step 4: Length-limit to HUFF_MAX_BITS (package-merge would be ideal,
    // but simple clamping + canonical rebuild works for our purposes)
    build_canonical_codes(lengths);
}

// ─────────────────────────────────────────────
//  BUILD CANONICAL CODES FROM LENGTHS
// ─────────────────────────────────────────────

void HuffmanCoder::build_canonical_codes(uint8_t lengths[HUFF_SYMBOLS])
{
    std::memset(codes_, 0, sizeof(codes_));

    // Count symbols per length
    int bl_count[HUFF_MAX_BITS + 1] = {};
    for (int s = 0; s < HUFF_SYMBOLS; ++s)
        if (lengths[s] > 0)
            bl_count[lengths[s]]++;

    // Find starting code for each length
    uint16_t next_code[HUFF_MAX_BITS + 1] = {};
    uint16_t code = 0;
    for (int bits = 1; bits <= HUFF_MAX_BITS; ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes: symbols with same length get consecutive codes
    // Sort symbols by (length, symbol_value) for determinism
    int order[HUFF_SYMBOLS];
    for (int i = 0; i < HUFF_SYMBOLS; ++i) order[i] = i;
    std::stable_sort(order, order + HUFF_SYMBOLS, [&](int a, int b){
        if (lengths[a] != lengths[b]) return lengths[a] < lengths[b];
        return a < b;
    });

    for (int i = 0; i < HUFF_SYMBOLS; ++i) {
        int s = order[i];
        if (lengths[s] > 0) {
            codes_[s].length = lengths[s];
            codes_[s].code   = next_code[lengths[s]]++;
        }
    }

    build_decode_table();
}

// ─────────────────────────────────────────────
//  BUILD O(1) DECODE TABLE
// ─────────────────────────────────────────────

void HuffmanCoder::build_decode_table()
{
    int table_size = 1 << HUFF_MAX_BITS;
    decode_table_.assign(table_size, {0, 0});

    for (int s = 0; s < HUFF_SYMBOLS; ++s) {
        if (codes_[s].length == 0) continue;

        // Fill all table entries that match this code
        // A code of length L matches any (HUFF_MAX_BITS - L) suffix
        int len    = codes_[s].length;
        int c      = codes_[s].code;
        int spread = 1 << (HUFF_MAX_BITS - len);

        // The code occupies the top `len` bits of the table index
        int base = c << (HUFF_MAX_BITS - len);
        for (int j = 0; j < spread; ++j) {
            decode_table_[base + j] = {(uint8_t)s, (uint8_t)len};
        }
    }
}

// ─────────────────────────────────────────────
//  ENCODE
// ─────────────────────────────────────────────

std::vector<uint8_t> HuffmanCoder::encode(const uint8_t* data, size_t size) const
{
    std::vector<uint8_t> out;
    out.reserve(size);  // output will be <= input size for typical data

    uint32_t bit_buf = 0;   // bit accumulator
    int      bit_cnt = 0;   // bits in accumulator

    for (size_t i = 0; i < size; ++i) {
        const HuffCode& hc = codes_[data[i]];
        if (hc.length == 0)
            throw std::runtime_error("Huffman: symbol has no code");

        // Shift code into bit buffer (MSB first)
        bit_buf = (bit_buf << hc.length) | hc.code;
        bit_cnt += hc.length;

        // Flush complete bytes
        while (bit_cnt >= 8) {
            bit_cnt -= 8;
            out.push_back((bit_buf >> bit_cnt) & 0xFF);
        }
    }

    // Flush remaining bits with zero padding
    if (bit_cnt > 0) {
        uint8_t padding = (uint8_t)(8 - bit_cnt);
        out.push_back((bit_buf << padding) & 0xFF);
        last_padding_ = padding;
    } else {
        last_padding_ = 0;
    }

    return out;
}

// ─────────────────────────────────────────────
//  DECODE
// ─────────────────────────────────────────────

std::vector<uint8_t> HuffmanCoder::decode(const uint8_t* bits, size_t bits_size,
                                           uint8_t padding, size_t out_size) const
{
    std::vector<uint8_t> out;
    out.reserve(out_size);

    uint32_t bit_buf = 0;
    int      bit_cnt = 0;
    size_t   total_bits = bits_size * 8 - padding;
    size_t   bits_consumed = 0;

    int table_bits = HUFF_MAX_BITS;
    size_t byte_idx = 0;

    while (bits_consumed < total_bits) {
        // Fill bit buffer from input bytes
        while (bit_cnt < table_bits && byte_idx < bits_size) {
            bit_buf = (bit_buf << 8) | bits[byte_idx++];
            bit_cnt += 8;
        }

        if (bit_cnt == 0) break;

        // Peek top HUFF_MAX_BITS bits for O(1) table lookup
        int peek_bits = std::min(bit_cnt, table_bits);
        int shift = bit_cnt - peek_bits;
        int idx = (bit_buf >> shift) & ((1 << peek_bits) - 1);

        // Pad to table width if fewer bits available
        if (peek_bits < table_bits)
            idx <<= (table_bits - peek_bits);

        const DecodeEntry& entry = decode_table_[idx];
        if (entry.length == 0 || entry.length > bit_cnt)
            break;  // end of valid data

        out.push_back(entry.symbol);
        bit_cnt -= entry.length;
        bits_consumed += entry.length;
        bit_buf &= (1u << bit_cnt) - 1;  // clear consumed bits

        if (out_size > 0 && out.size() >= out_size)
            break;
    }

    return out;
}

// ─────────────────────────────────────────────
//  SERIALIZE / DESERIALIZE CODE LENGTHS
// ─────────────────────────────────────────────

void HuffmanCoder::serialize_lengths(uint8_t out[HUFF_SYMBOLS]) const
{
    for (int s = 0; s < HUFF_SYMBOLS; ++s)
        out[s] = codes_[s].length;
}

void HuffmanCoder::deserialize_lengths(const uint8_t lengths[HUFF_SYMBOLS])
{
    uint8_t tmp[HUFF_SYMBOLS];
    std::memcpy(tmp, lengths, HUFF_SYMBOLS);
    build_canonical_codes(tmp);
}

} // namespace clm
