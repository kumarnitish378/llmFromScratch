/**
 * ============================================================
 *  clm_demo.h / clm_demo.cpp
 *  CLM Compressor — CLI, Tests, Benchmark
 *
 *  This file contains ALL demo/test/CLI logic.
 *  Your main.cpp just calls the functions you need.
 *
 *  PUBLIC API:
 *    clm::demo::compress_file(path)       compress a file
 *    clm::demo::decompress_file(path)     decompress a .clm file
 *    clm::demo::run_tests()               run built-in test suite
 *    clm::demo::benchmark(path)           benchmark on a file
 *    clm::demo::run_cli(argc, argv)       full CLI handler
 * ============================================================
 */

#pragma once

#include "../libraries/CLM_Compressor/compressor.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace clm {
namespace demo {

// ─────────────────────────────────────────────
//  INTERNAL HELPERS (not exposed)
// ─────────────────────────────────────────────

namespace detail {

inline void sep(char c = '-', int w = 62) {
    std::cout << "  " << std::string(w, c) << "\n";
}

inline void header(const std::string& title) {
    std::cout << "\n"; sep('=');
    std::cout << "  " << title << "\n";
    sep('=');
}

inline std::string fmt_bytes(uint64_t b) {
    std::ostringstream ss;
    if      (b >= 1024*1024) ss << std::fixed << std::setprecision(2) << b/(1024.0*1024.0) << " MB";
    else if (b >= 1024)      ss << std::fixed << std::setprecision(2) << b/1024.0           << " KB";
    else                     ss << b << " B";
    return ss.str();
}

inline std::string fmt_ms(double ms) {
    std::ostringstream ss;
    if (ms >= 1000.0) ss << std::fixed << std::setprecision(2) << ms/1000.0 << " s";
    else              ss << std::fixed << std::setprecision(1) << ms         << " ms";
    return ss.str();
}

inline void print_stats(const CompressionStats& s) {
    sep();
    std::cout << "  Original size    : " << std::setw(14) << fmt_bytes(s.original_bytes)   << "\n"
              << "  Compressed size  : " << std::setw(14) << fmt_bytes(s.compressed_bytes) << "\n";
    sep();
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(4)
              << std::setw(10) << s.ratio << "x\n"
              << "  Space saving     : " << std::setw(10) << std::setprecision(1)
              << s.saving_pct << "%  "
              << (s.saving_pct > 0 ? "saved" : "overhead") << "\n";
    sep();
    std::cout << "  Chunks           : " << std::setw(14) << s.num_chunks    << "\n"
              << "  Threads used     : " << std::setw(14) << s.threads_used  << "\n"
              << "  Compress time    : " << std::setw(14) << fmt_ms(s.compress_ms) << "\n";
    if (s.decompress_ms > 0)
        std::cout << "  Decompress time  : " << std::setw(14) << fmt_ms(s.decompress_ms) << "\n";
    if (s.compress_ms > 0) {
        double mbps = (s.original_bytes / (1024.0*1024.0)) / (s.compress_ms / 1000.0);
        std::cout << "  Throughput       : " << std::setw(10) << std::setprecision(1)
                  << mbps << " MB/s\n";
    }
    sep();
}

} // namespace detail

// ─────────────────────────────────────────────
//  PUBLIC: COMPRESS FILE
// ─────────────────────────────────────────────

/**
 * Compress input_path → <name>_compressed.clm
 * Prints stats to stdout.
 */
inline CompressionStats compress_file(const std::string& input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string out_path = (p.parent_path() / (p.stem().string() + "_compressed")).string() + ".clm";

    detail::header("COMPRESS: " + input_path);
    std::cout << "  Output  : " << out_path << "\n"
              << "  Threads : " << std::thread::hardware_concurrency() << " (auto)\n\n";

    CompressorConfig cfg;
    cfg.verbose = true;
    Compressor comp(cfg);

    auto stats = comp.compress_file(input_path, out_path);
    std::cout << "\n";
    detail::print_stats(stats);
    std::cout << "  Saved -> " << out_path << "\n\n";
    return stats;
}

// ─────────────────────────────────────────────
//  PUBLIC: DECOMPRESS FILE
// ─────────────────────────────────────────────

/**
 * Decompress input_path (.clm) → <name>_decompressed.txt
 */
inline void decompress_file(const std::string& input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    if (stem.size() > 11 && stem.substr(stem.size()-11) == "_compressed")
        stem = stem.substr(0, stem.size()-11);
    std::string out_path = (p.parent_path() / (stem + "_decompressed.txt")).string();

    detail::header("DECOMPRESS: " + input_path);
    std::cout << "  Output : " << out_path << "\n\n";

    CompressorConfig cfg; cfg.verbose = true;
    Compressor comp(cfg);
    comp.decompress_file(input_path, out_path);

    std::cout << "  Decompress time: "
              << detail::fmt_ms(comp.last_stats().decompress_ms) << "\n"
              << "  Saved -> " << out_path << "\n\n";
}

// ─────────────────────────────────────────────
//  PUBLIC: RUN TESTS
// ─────────────────────────────────────────────

/**
 * Runs built-in correctness + multithreading tests.
 * Returns true if all passed.
 */
inline bool run_tests() {
    detail::header("BUILT-IN TEST SUITE");

    struct TC { std::string label, text; };
    std::vector<TC> cases = {
        {"Empty string",      ""},
        {"Single char",       "A"},
        {"Short repeated",    "abababababababababababab"},
        {"English sentence",  "The quick brown fox jumps over the lazy dog."},
        {"Code-like",         "for(int i=0;i<n;i++){sum+=arr[i];}\nreturn sum;\n"},
        {"Highly repetitive", std::string(10000,'A') + std::string(10000,'B')},
        {"Natural language",
            "In the beginning was the word and the word was with the compiler. "
            "All programs are created equal but some are more equal than others. "
            "In the beginning was the word and the word was with the compiler."},
        {"Mixed content",
            "SELECT * FROM tokens WHERE freq > 100 ORDER BY freq DESC;\n"
            "SELECT * FROM tokens WHERE freq > 100 ORDER BY freq DESC;\n"
            "def train(model):\n    for epoch in range(100):\n        loss.backward()\n"},
    };

    CompressorConfig cfg; cfg.verbose = false;
    Compressor comp(cfg);
    int passed = 0, failed = 0;

    std::cout << "\n  " << std::left
              << std::setw(24) << "Test"
              << std::setw(10) << "Input"
              << std::setw(12) << "Compressed"
              << std::setw(8)  << "Ratio"
              << "Status\n";
    detail::sep();

    for (auto& tc : cases) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(tc.text.data());
        size_t sz = tc.text.size();
        bool ok = false;
        std::vector<uint8_t> compressed;

        try {
            if (sz == 0) { ok = true; compressed = {0}; }
            else {
                compressed       = comp.compress(ptr, sz);
                auto decompressed = comp.decompress(compressed.data(), compressed.size());
                ok = (decompressed.size() == sz &&
                      std::memcmp(decompressed.data(), ptr, sz) == 0);
            }
        } catch (const std::exception& e) {
            std::cerr << "  EXCEPTION: " << e.what() << "\n";
        }

        double ratio = sz > 0 ? (double)compressed.size() / sz : 1.0;
        std::cout << "  " << std::left
                  << std::setw(24) << tc.label.substr(0,23)
                  << std::setw(10) << detail::fmt_bytes(sz)
                  << std::setw(12) << detail::fmt_bytes(compressed.size())
                  << std::setw(8)  << std::fixed << std::setprecision(3) << ratio
                  << (ok ? "PASS" : "FAIL") << "\n";

        ok ? ++passed : ++failed;
    }

    // Multithreading stress test
    detail::sep();
    std::cout << "\n  MULTITHREADING STRESS TEST  (2 MB repetitive text)\n";
    detail::sep();

    std::string big(2*1024*1024, '\0');
    const std::string pat = "the quick brown fox jumped over the lazy dog. ";
    for (size_t i = 0; i < big.size(); ++i) big[i] = pat[i % pat.size()];

    for (uint32_t t : {1u, 2u, 4u, std::max(1u, std::thread::hardware_concurrency())}) {
        CompressorConfig tcfg; tcfg.num_threads = t; tcfg.verbose = false;
        Compressor tc(tcfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto cd  = tc.compress(reinterpret_cast<const uint8_t*>(big.data()), big.size());
        auto dd  = tc.decompress(cd.data(), cd.size());
        double ms = std::chrono::duration<double,std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
        bool ok = (dd.size() == big.size() &&
                   std::memcmp(dd.data(), big.data(), big.size()) == 0);
        double mbps = (big.size()/(1024.0*1024.0)) / (ms/1000.0);
        ok ? ++passed : ++failed;
        std::cout << "  threads=" << std::setw(2) << t
                  << "  ratio=" << std::fixed << std::setprecision(4) << (double)cd.size()/big.size()
                  << "  time=" << std::setw(8) << detail::fmt_ms(ms)
                  << "  " << std::setprecision(1) << mbps << " MB/s"
                  << "  " << (ok ? "PASS" : "FAIL") << "\n";
    }

    detail::sep();
    std::cout << "\n  " << passed << " passed, " << failed << " failed  "
              << (failed == 0 ? "-- ALL PASSED" : "-- SOME FAILED") << "\n";
    return failed == 0;
}

// ─────────────────────────────────────────────
//  PUBLIC: BENCHMARK
// ─────────────────────────────────────────────

/**
 * Benchmark compress + decompress on a file across thread counts.
 */
inline void benchmark(const std::string& input_path) {
    detail::header("BENCHMARK: " + input_path);

    std::ifstream f(input_path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "  Cannot open file\n"; return; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);

    std::cout << "  File: " << detail::fmt_bytes(sz)
              << "  |  HW threads: " << std::thread::hardware_concurrency() << "\n\n"
              << "  " << std::left
              << std::setw(10) << "Threads"
              << std::setw(14) << "Compressed"
              << std::setw(8)  << "Ratio"
              << std::setw(14) << "Compress"
              << std::setw(14) << "Decompress"
              << "Throughput\n";
    detail::sep();

    for (uint32_t t : {1u, 2u, 4u, std::max(1u, std::thread::hardware_concurrency())}) {
        CompressorConfig cfg; cfg.num_threads = t; cfg.verbose = false;
        Compressor comp(cfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto cd  = comp.compress(data.data(), sz);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dd  = comp.decompress(cd.data(), cd.size());
        auto t2 = std::chrono::high_resolution_clock::now();
        double c_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        double d_ms = std::chrono::duration<double,std::milli>(t2-t1).count();
        bool ok = (dd.size()==sz && std::memcmp(dd.data(),data.data(),sz)==0);

        std::cout << "  " << std::left
                  << std::setw(10) << t
                  << std::setw(14) << detail::fmt_bytes(cd.size())
                  << std::setw(8)  << std::fixed << std::setprecision(4) << (double)cd.size()/sz
                  << std::setw(14) << detail::fmt_ms(c_ms)
                  << std::setw(14) << detail::fmt_ms(d_ms)
                  << std::setprecision(1) << (sz/(1024.0*1024.0))/(c_ms/1000.0) << " MB/s"
                  << (ok ? "" : "  MISMATCH!") << "\n";
    }
    detail::sep();
    std::cout << "\n";
}

// ─────────────────────────────────────────────
//  PUBLIC: FULL CLI HANDLER
// ─────────────────────────────────────────────

/**
 * Drop-in CLI handler. Call from your main() if you want the full CLI.
 * Returns exit code (0 = success).
 *
 * Usage:
 *   ./app <file>           compress
 *   ./app -d <file.clm>    decompress
 *   ./app --test           run tests
 *   ./app --bench <file>   benchmark
 */
inline int run_cli(int argc, char* argv[]) {
    std::cout << "+------------------------------------------------------+\n"
              << "|  CLM Compressor  LZ77 + Huffman + Multithreading    |\n"
              << "+------------------------------------------------------+\n";

    auto usage = [](){
        std::cout << "\n  Usage:\n"
                  << "    <file>           Compress -> <name>_compressed.clm\n"
                  << "    -d <file.clm>    Decompress -> <name>_decompressed.txt\n"
                  << "    --test           Run test suite\n"
                  << "    --bench <file>   Benchmark across thread counts\n\n";
    };

    if (argc == 1)               { usage(); return run_tests() ? 0 : 1; }
    std::string a = argv[1];
    if (a == "--test")           { return run_tests() ? 0 : 1; }
    if (a == "-d"   && argc>=3)  { decompress_file(argv[2]); return 0; }
    if (a == "--bench" && argc>=3){ benchmark(argv[2]); return 0; }
    if (a[0] != '-')             { compress_file(a); return 0; }
    usage(); return 1;
}

} // namespace demo
} // namespace clm
