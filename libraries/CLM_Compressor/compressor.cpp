/**
 * compressor.cpp — Multithreaded Chunked Compression Pipeline
 */

#include "compressor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace clm {

// Magic bytes for file format identification
static const uint8_t FILE_MAGIC[8] = {'C','L','M',0x01,0x00,0x00,0x00,0x00};

// ─────────────────────────────────────────────
//  CONSTRUCTOR
// ─────────────────────────────────────────────

Compressor::Compressor(CompressorConfig cfg)
    : cfg_(cfg)
{
    num_threads_ = cfg_.num_threads;
    if (num_threads_ == 0)
        num_threads_ = CLM_HW_THREADS();
}

// ─────────────────────────────────────────────
//  COMPRESS ONE CHUNK  (thread-safe, no shared state)
// ─────────────────────────────────────────────

CompressedChunk Compressor::compress_chunk(const uint8_t* data, size_t size)
{
    // Each thread has its own LZ77Compressor (its hash tables are instance vars)
    LZ77Compressor lz77;

    // Stage 1: LZ77
    auto tokens = lz77.compress(data, size, cfg_.lazy_match);

    // Stage 2: Serialize tokens to byte stream
    auto serialized = LZ77Compressor::serialize(tokens);

    // Stage 3: Count byte frequencies for Huffman
    uint64_t freq[HUFF_SYMBOLS] = {};
    for (uint8_t b : serialized)
        freq[b]++;

    // Stage 4: Build Huffman codes
    HuffmanCoder huff;
    huff.build(freq);

    // Stage 5: Huffman encode
    auto compressed = huff.encode(serialized.data(), serialized.size());

    // Package result
    CompressedChunk chunk;
    chunk.orig_size = (uint32_t)size;
    chunk.padding   = huff.last_padding();
    huff.serialize_lengths(chunk.huff_lengths);
    chunk.data = std::move(compressed);

    return chunk;
}

// ─────────────────────────────────────────────
//  DECOMPRESS ONE CHUNK
// ─────────────────────────────────────────────

std::vector<uint8_t> Compressor::decompress_chunk(const CompressedChunk& chunk)
{
    // Stage 1: Huffman decode
    HuffmanCoder huff;
    huff.deserialize_lengths(chunk.huff_lengths);

    auto serialized = huff.decode(chunk.data.data(), chunk.data.size(),
                                  chunk.padding, 0);

    // Stage 2: Deserialize LZ77 tokens
    auto tokens = LZ77Compressor::deserialize(serialized.data(), serialized.size());

    // Stage 3: LZ77 decompress
    LZ77Compressor lz77;
    return lz77.decompress(tokens, chunk.orig_size);
}

// ─────────────────────────────────────────────
//  SERIALIZE CHUNKS → FILE FORMAT BYTES
// ─────────────────────────────────────────────

std::vector<uint8_t> Compressor::serialize_chunks(
    const std::vector<CompressedChunk>& chunks, uint64_t orig_size)
{
    // Calculate total output size
    size_t total = 8 + 8 + 4 + 4;  // magic + orig_size + chunk_count + chunk_size
    for (auto& c : chunks)
        total += 4 + 4 + 1 + HUFF_SYMBOLS + c.data.size();

    std::vector<uint8_t> out;
    out.reserve(total);

    auto write_u8  = [&](uint8_t v)  { out.push_back(v); };
    auto write_u32 = [&](uint32_t v) {
        out.push_back(v & 0xFF);
        out.push_back((v >> 8)  & 0xFF);
        out.push_back((v >> 16) & 0xFF);
        out.push_back((v >> 24) & 0xFF);
    };
    auto write_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i)
            out.push_back((v >> (i * 8)) & 0xFF);
    };

    // Header
    for (uint8_t b : FILE_MAGIC) write_u8(b);
    write_u64(orig_size);
    write_u32((uint32_t)chunks.size());
    write_u32(cfg_.chunk_size);

    // Chunks
    for (auto& c : chunks) {
        write_u32((uint32_t)c.data.size());   // compressed size
        write_u32(c.orig_size);               // original size of this chunk
        write_u8(c.padding);                  // huffman padding
        // Huffman code lengths (256 bytes)
        for (int s = 0; s < HUFF_SYMBOLS; ++s)
            write_u8(c.huff_lengths[s]);
        // Compressed data
        out.insert(out.end(), c.data.begin(), c.data.end());
    }

    return out;
}

// ─────────────────────────────────────────────
//  PARSE FILE FORMAT → CHUNKS
// ─────────────────────────────────────────────

std::vector<CompressedChunk> Compressor::parse_chunks(
    const uint8_t* data, size_t size,
    uint64_t& orig_size_out, uint32_t& chunk_size_out)
{
    if (size < 24)
        throw std::runtime_error("Compressor: file too small to be valid");

    // Verify magic
    if (std::memcmp(data, FILE_MAGIC, 8) != 0)
        throw std::runtime_error("Compressor: invalid file magic (not a CLM file)");

    size_t pos = 8;
    auto read_u8  = [&]() -> uint8_t  { return data[pos++]; };
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v = (uint32_t)data[pos]
                   | ((uint32_t)data[pos+1] << 8)
                   | ((uint32_t)data[pos+2] << 16)
                   | ((uint32_t)data[pos+3] << 24);
        pos += 4;
        return v;
    };
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= ((uint64_t)data[pos++] << (i * 8));
        return v;
    };

    orig_size_out  = read_u64();
    uint32_t n_chunks = read_u32();
    chunk_size_out = read_u32();

    std::vector<CompressedChunk> chunks(n_chunks);
    for (uint32_t i = 0; i < n_chunks; ++i) {
        uint32_t comp_size = read_u32();
        chunks[i].orig_size = read_u32();
        chunks[i].padding   = read_u8();

        for (int s = 0; s < HUFF_SYMBOLS; ++s)
            chunks[i].huff_lengths[s] = read_u8();

        chunks[i].data.assign(data + pos, data + pos + comp_size);
        pos += comp_size;
    }

    return chunks;
}

// ─────────────────────────────────────────────
//  COMPRESS (in-memory, multithreaded)
// ─────────────────────────────────────────────

std::vector<uint8_t> Compressor::compress(const uint8_t* data, size_t size)
{
    // Split input into chunks
    uint32_t n_chunks = (uint32_t)((size + cfg_.chunk_size - 1) / cfg_.chunk_size);
    if (n_chunks == 0) n_chunks = 1;

    std::vector<CompressedChunk> results(n_chunks);
    std::atomic<uint32_t> chunks_done{0};
    std::atomic<uint32_t> next_chunk{0};

    // Thread worker function
    auto worker = [&]() {
        while (true) {
            uint32_t idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n_chunks) return;

            size_t offset = (size_t)idx * cfg_.chunk_size;
            size_t chunk_sz = std::min((size_t)cfg_.chunk_size, size - offset);

            results[idx] = compress_chunk(data + offset, chunk_sz);

            uint32_t done = chunks_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (cfg_.verbose && cfg_.progress_cb)
                cfg_.progress_cb(done, n_chunks);
        }
    };

    // Launch threads (cap at n_chunks — no point more threads than work)
    uint32_t t = std::min(num_threads_, n_chunks);
    std::vector<CLM_THREAD_T> threads;
    threads.reserve(t);
    for (uint32_t i = 0; i < t; ++i)
        threads.push_back(CLM_START_THREAD(worker));
    for (auto& th : threads)
        th.join();

    return serialize_chunks(results, (uint64_t)size);
}

// ─────────────────────────────────────────────
//  DECOMPRESS (in-memory, multithreaded)
// ─────────────────────────────────────────────

std::vector<uint8_t> Compressor::decompress(const uint8_t* data, size_t size)
{
    uint64_t orig_size;
    uint32_t chunk_size;
    auto chunks = parse_chunks(data, size, orig_size, chunk_size);

    uint32_t n_chunks = (uint32_t)chunks.size();
    std::vector<std::vector<uint8_t>> results(n_chunks);
    std::atomic<uint32_t> next_chunk{0};

    auto worker = [&]() {
        while (true) {
            uint32_t idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n_chunks) return;
            results[idx] = decompress_chunk(chunks[idx]);
        }
    };

    uint32_t t = std::min(num_threads_, n_chunks);
    std::vector<CLM_THREAD_T> threads;
    threads.reserve(t);
    for (uint32_t i = 0; i < t; ++i)
        threads.push_back(CLM_START_THREAD(worker));
    for (auto& th : threads)
        th.join();

    // Concatenate results in order
    std::vector<uint8_t> out;
    out.reserve(orig_size);
    for (auto& r : results)
        out.insert(out.end(), r.begin(), r.end());

    return out;
}

// ─────────────────────────────────────────────
//  FILE I/O
// ─────────────────────────────────────────────

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write file: " + path);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

CompressionStats Compressor::compress_file(const std::string& input_path,
                                            const std::string& output_path)
{
    auto input = read_file(input_path);
    stats_ = {};
    stats_.original_bytes = input.size();
    stats_.threads_used   = num_threads_;
    // Setup progress reporting
    uint32_t n_chunks = (uint32_t)((input.size() + cfg_.chunk_size - 1) / cfg_.chunk_size);
    if (cfg_.verbose) {
        cfg_.progress_cb = [&](uint32_t done, uint32_t total) {
            std::cout << "\r  Compressing... " << done << "/" << total
                      << " chunks (" << (done * 100 / total) << "%)" << std::flush;
        };
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto output = compress(input.data(), input.size());
    auto t1 = std::chrono::high_resolution_clock::now();

    if (cfg_.verbose) std::cout << "\n";

    write_file(output_path, output);

    stats_.compressed_bytes = output.size();
    stats_.num_chunks       = n_chunks;
    stats_.ratio            = (double)output.size() / input.size();
    stats_.saving_pct       = (1.0 - stats_.ratio) * 100.0;
    stats_.compress_ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return stats_;
}

void Compressor::decompress_file(const std::string& input_path,
                                  const std::string& output_path)
{
    auto input = read_file(input_path);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto output = decompress(input.data(), input.size());
    auto t1 = std::chrono::high_resolution_clock::now();

    write_file(output_path, output);
    stats_.decompress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace clm
