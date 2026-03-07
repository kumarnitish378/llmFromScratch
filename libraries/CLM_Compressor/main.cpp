/**
 * main.cpp — llmFromScratch
 * Your existing code stays here. Add the compressor lines you need.
 */

// ── Your existing includes ────────────────────────────────
// #include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"
// #include "libraries/NKS_Tokenizer/NKS_SentencePieceTokenizer.h"

// ── Add this one line for compression ────────────────────
#include "libraries/CLM_Compressor/CLM_Compressor.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main() {

    // ── Example 1: Compress your training data file ───────
    {
        clm::CompressorConfig cfg;
        cfg.num_threads = 0;       // auto — uses all CPU cores
        cfg.verbose     = true;

        clm::Compressor comp(cfg);

        std::cout << "Compressing Training_Essay_Data.txt ...\n";
        auto stats = comp.compress_file(
            "Data/Training_Essay_Data.txt",
            "Data/Training_Essay_Data.clm"
        );

        std::cout << "Original : " << stats.original_bytes   << " bytes\n"
                  << "Compressed: " << stats.compressed_bytes << " bytes\n"
                  << "Ratio    : " << stats.ratio             << "\n"
                  << "Saving   : " << stats.saving_pct        << "%\n"
                  << "Speed    : " << stats.compress_ms       << " ms\n";
    }

    // ── Example 2: Compress text in memory (LLM pipeline) ─
    {
        std::string text = "The quick brown fox jumps over the lazy dog. "
                           "The quick brown fox jumps over the lazy dog.";

        clm::Compressor comp;

        // Compress string → bytes
        std::vector<uint8_t> raw(text.begin(), text.end());
        auto compressed   = comp.compress(raw.data(), raw.size());
        auto decompressed = comp.decompress(compressed.data(), compressed.size());

        // Convert back to string
        std::string restored(decompressed.begin(), decompressed.end());

        std::cout << "\nIn-memory roundtrip: "
                  << (restored == text ? "PASS" : "FAIL") << "\n"
                  << "Original  : " << raw.size()        << " bytes\n"
                  << "Compressed: " << compressed.size() << " bytes\n";
    }

    return 0;
}
