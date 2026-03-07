#pragma once
/**
 * ============================================================
 *  compressor.h  —  Multithreaded Chunked Compression Pipeline
 *  CompressedLLM / General purpose text compression
 * ============================================================
 *
 *  PIPELINE (per chunk):
 *    Raw bytes → LZ77 tokens → Serialize tokens → Huffman encode → Chunk
 *
 *  MULTITHREADING STRATEGY:
 *    - Input is split into fixed-size chunks (default 512KB each)
 *    - Each chunk is compressed independently by a worker thread
 *    - Output chunks are written in-order after all threads complete
 *    - No inter-chunk dependencies (unlike sliding-window across chunks)
 *
 *  WHY CHUNKED (not one giant LZ77 pass):
 *    - Parallelism: N chunks → N threads → ~N× speedup on multi-core
 *    - Memory: each thread uses ~2MB working memory vs 1.2GB Python
 *    - Seekability: random access to any chunk for future streaming use
 *
 *  TRADEOFF:
 *    Back-references cannot cross chunk boundaries.
 *    Slightly lower ratio than single-pass for highly repetitive inputs.
 *    For natural language/code this is negligible (<2% ratio difference).
 *
 *  FILE FORMAT:
 *    [8B]  Magic:          "CLM\x01\x00\x00\x00\x00"
 *    [8B]  Original size:  uint64_t (little-endian)
 *    [4B]  Chunk count:    uint32_t
 *    [4B]  Chunk size:     uint32_t (uncompressed bytes per chunk)
 *    For each chunk:
 *      [4B]  Compressed size of this chunk (uint32_t)
 *      [4B]  Original size of this chunk   (uint32_t)
 *      [1B]  Huffman padding bits
 *      [256B] Huffman code lengths (canonical)
 *      [NB]  Compressed data
 * ============================================================
 */

#ifndef CLM_COMPRESSOR_H
#define CLM_COMPRESSOR_H

#include "lz77.h"
#include "huffman.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <atomic>

// ── Cross-platform threading ──────────────────────────────────────────────
// MinGW < 9 ships a broken std::thread on Windows.
// We detect Win32 and use native _beginthreadex instead.
#if defined(_WIN32)
#  include <windows.h>
#  include <process.h>
   namespace clm { namespace detail {
       struct NativeThread {
           HANDLE handle = nullptr;
           NativeThread() = default;
           NativeThread(NativeThread&& o) noexcept : handle(o.handle) { o.handle = nullptr; }
           NativeThread(const NativeThread&) = delete;
           ~NativeThread() { if (handle) CloseHandle(handle); }
           void join() {
               if (handle) {
                   WaitForSingleObject(handle, INFINITE);
                   CloseHandle(handle);
                   handle = nullptr;
               }
           }
       };
       inline unsigned __stdcall thread_entry(void* arg) {
           auto* fn = static_cast<std::function<void()>*>(arg);
           (*fn)(); delete fn; return 0;
       }
       inline NativeThread start_thread(std::function<void()> fn) {
           NativeThread t;
           auto* heap_fn = new std::function<void()>(std::move(fn));
           t.handle = reinterpret_cast<HANDLE>(
               _beginthreadex(nullptr, 0, thread_entry, heap_fn, 0, nullptr));
           return t;
       }
       inline unsigned int hardware_threads() {
           SYSTEM_INFO si; GetSystemInfo(&si);
           return static_cast<unsigned int>(si.dwNumberOfProcessors);
       }
   }} // namespace clm::detail
#  define CLM_THREAD_T          clm::detail::NativeThread
#  define CLM_START_THREAD(fn)  clm::detail::start_thread(fn)
#  define CLM_HW_THREADS()      clm::detail::hardware_threads()
#else
#  include <thread>
#  define CLM_THREAD_T          std::thread
#  define CLM_START_THREAD(fn)  std::thread(fn)
#  define CLM_HW_THREADS()      std::max(1u, std::thread::hardware_concurrency())
#endif
// ─────────────────────────────────────────────────────────────────────────

namespace clm {

// ─────────────────────────────────────────────
//  CONFIG
// ─────────────────────────────────────────────

struct CompressorConfig {
    uint32_t chunk_size   = 512 * 1024;          // 512KB per chunk
    uint32_t num_threads  = 0;                    // 0 = auto-detect (hardware_concurrency)
    bool     lazy_match   = true;                 // LZ77 lazy matching
    bool     verbose      = false;                // print progress

    // Progress callback: called after each chunk completes
    // args: (chunks_done, total_chunks)
    std::function<void(uint32_t, uint32_t)> progress_cb;
};

// ─────────────────────────────────────────────
//  COMPRESSED CHUNK (output unit)
// ─────────────────────────────────────────────

struct CompressedChunk {
    uint32_t              orig_size;       // uncompressed size of this chunk
    uint8_t               padding;         // Huffman padding bits
    uint8_t               huff_lengths[HUFF_SYMBOLS];  // canonical code lengths
    std::vector<uint8_t>  data;            // compressed bytes
};

// ─────────────────────────────────────────────
//  STATS (returned after compression)
// ─────────────────────────────────────────────

struct CompressionStats {
    uint64_t original_bytes   = 0;
    uint64_t compressed_bytes = 0;
    uint32_t num_chunks       = 0;
    uint32_t threads_used     = 0;
    double   ratio            = 0.0;
    double   saving_pct       = 0.0;
    double   compress_ms      = 0.0;
    double   decompress_ms    = 0.0;
    uint64_t lz77_literals    = 0;
    uint64_t lz77_refs        = 0;
};

// ─────────────────────────────────────────────
//  MAIN COMPRESSOR CLASS
// ─────────────────────────────────────────────

class Compressor {
public:
    explicit Compressor(CompressorConfig cfg = {});

    /**
     * Compress raw bytes in memory.
     * Returns serialized compressed bytes (full file format).
     */
    std::vector<uint8_t> compress(const uint8_t* data, size_t size);

    /**
     * Decompress previously compressed bytes.
     */
    std::vector<uint8_t> decompress(const uint8_t* data, size_t size);

    /**
     * Compress a file → write to output path.
     * Returns stats.
     */
    CompressionStats compress_file(const std::string& input_path,
                                   const std::string& output_path);

    /**
     * Decompress a file → write to output path.
     */
    void decompress_file(const std::string& input_path,
                         const std::string& output_path);

    const CompressionStats& last_stats() const { return stats_; }

private:
    CompressorConfig  cfg_;
    CompressionStats  stats_;
    uint32_t          num_threads_;

    // Compress one chunk (called by worker threads)
    CompressedChunk compress_chunk(const uint8_t* data, size_t size);

    // Decompress one chunk
    std::vector<uint8_t> decompress_chunk(const CompressedChunk& chunk);

    // Serialize all chunks to final byte stream
    std::vector<uint8_t> serialize_chunks(const std::vector<CompressedChunk>& chunks,
                                           uint64_t orig_size);

    // Parse byte stream back to chunks
    std::vector<CompressedChunk> parse_chunks(const uint8_t* data, size_t size,
                                               uint64_t& orig_size_out,
                                               uint32_t& chunk_size_out);
};

} // namespace clm

#endif // CLM_COMPRESSOR_H
