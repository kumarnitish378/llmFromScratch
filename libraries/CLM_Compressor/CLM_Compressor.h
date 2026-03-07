#pragma once
/**
 * ============================================================
 *  CLM_Compressor.h  —  Single include for your project
 *  Drop into: libraries/CLM_Compressor/CLM_Compressor.h
 * ============================================================
 *
 *  USAGE IN YOUR main.cpp:
 *    #include "libraries/CLM_Compressor/CLM_Compressor.h"
 *
 *  QUICK API:
 *
 *    // --- Compress / Decompress files ---
 *    clm::Compressor comp;
 *    comp.compress_file("Data/Training_Essay_Data.txt", "Data/essays.clm");
 *    comp.decompress_file("Data/essays.clm", "Data/essays_restored.txt");
 *
 *    // --- Compress bytes in memory (for LLM pipeline) ---
 *    std::vector<uint8_t> raw(text.begin(), text.end());
 *    auto compressed   = comp.compress(raw.data(), raw.size());
 *    auto decompressed = comp.decompress(compressed.data(), compressed.size());
 *
 *    // --- Custom config ---
 *    clm::CompressorConfig cfg;
 *    cfg.num_threads = 8;         // 0 = auto detect
 *    cfg.chunk_size  = 512*1024;  // 512 KB per chunk
 *    cfg.verbose     = true;      // print progress
 *    clm::Compressor comp(cfg);
 *
 *    // --- Stats after compression ---
 *    auto stats = comp.last_stats();
 *    std::cout << "Ratio:  " << stats.ratio        << "\n";
 *    std::cout << "Speed:  " << stats.compress_ms  << " ms\n";
 *    std::cout << "Saving: " << stats.saving_pct   << "%\n";
 *
 * ============================================================
 *  COMPILE (add to your Makefile):
 *    SRCS += libraries/CLM_Compressor/lz77.cpp
 *    SRCS += libraries/CLM_Compressor/huffman.cpp
 *    SRCS += libraries/CLM_Compressor/compressor.cpp
 *    CXXFLAGS += -std=c++17 -O2 -pthread
 * ============================================================
 */

#include "lz77.h"
#include "huffman.h"
#include "compressor.h"
