#include "app_runner.h"

#include <cmath>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "libraries/NKS_Tokenizer/NKS_SentencePieceTokenizer.h"
#include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"
#include "libraries/CLM_Compressor/CLM_Compressor.h"
#include "libraries/NKS_LLM/NKS_LLM.h"

namespace {
std::string getEnvOrDefault(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::string(value);
}

struct AppPaths {
    std::string bpeTrainingPath = getEnvOrDefault("NKS_BPE_TRAINING_PATH", "Data/processed");
    std::string sentencePieceTrainingPath = getEnvOrDefault("NKS_SP_TRAINING_PATH", "Data/processed");
    std::string bpeModelPath = getEnvOrDefault("NKS_BPE_MODEL_PATH", "Metadata/bpe_model_processed.bin");
    std::string mergedTxtCorpusPath = getEnvOrDefault("NKS_MERGED_TXT_CORPUS_PATH", "Metadata/processed_txt_corpus.txt");
};

struct BpeRuntimeConfig {
    std::size_t mergeOps = 5000;
    std::size_t trainingWordLimit = 100000000;
    bool showTrainingProgress = true;
};

struct SentencePieceRuntimeConfig {
    std::size_t targetVocabSize = 2000;
    std::size_t maxPieceChars = 8;
    std::size_t trainingLineLimit = 25000;
};

enum class TokenizerMode {
    kBpe,
    kSentencePiece,
    kUnsupported
};

constexpr double kApproxCharsPerToken = 4.0;

struct TokenizationResult {
    std::vector<std::string> pieces;
    std::vector<int> tokenIds;
    std::string decodedText;
    std::size_t approxModelTokenCount = 0;
    std::size_t vocabularySize = 0;
};

void printBpeTrainingStats(const NKS_Tokenizer::TrainingStats& stats) {
    std::cout << "[BPE][Stats] rows=" << stats.rowsRead
              << ", tokens=" << stats.tokensUsedForTraining
              << ", unique_tokens=" << stats.uniqueTokens
              << ", final_vocab=" << stats.finalVocabSize << std::endl;
    std::cout << "[BPE][Stats] time_ms { load=" << static_cast<long long>(stats.loadMs)
              << ", train=" << static_cast<long long>(stats.trainMs)
              << ", build=" << static_cast<long long>(stats.buildMs)
              << " }" << std::endl;
}

bool endsWithIgnoreCase(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    const std::size_t offset = value.size() - suffix.size();
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(value[offset + i])));
        const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) {
            return false;
        }
    }
    return true;
}

#ifdef _WIN32
void collectTxtFilesRecursive(const std::string& directoryPath, std::vector<std::string>& files) {
    std::string pattern = directoryPath;
    if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
        pattern += "\\";
    }
    pattern += "*";

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        return;
    }

    do {
        const std::string name = findData.cFileName;
        if (name == "." || name == "..") {
            continue;
        }

        std::string fullPath = directoryPath;
        if (!fullPath.empty() && fullPath.back() != '\\' && fullPath.back() != '/') {
            fullPath += "\\";
        }
        fullPath += name;

        if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            collectTxtFilesRecursive(fullPath, files);
            continue;
        }

        if (endsWithIgnoreCase(name, ".txt")) {
            files.push_back(fullPath);
        }
    } while (FindNextFileA(hFind, &findData) != 0);

    FindClose(hFind);
}

std::vector<std::string> listTxtFilesInDirectory(const std::string& directoryPath) {
    std::vector<std::string> files;
    collectTxtFilesRecursive(directoryPath, files);
    std::sort(files.begin(), files.end());
    return files;
}
#else
void collectTxtFilesRecursive(const std::string& directoryPath, std::vector<std::string>& files) {
    DIR* dir = opendir(directoryPath.c_str());
    if (dir == nullptr) {
        return;
    }

    dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string fullPath = directoryPath;
        if (!fullPath.empty() && fullPath.back() != '/') {
            fullPath += "/";
        }
        fullPath += name;

        struct stat st;
        if (stat(fullPath.c_str(), &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            collectTxtFilesRecursive(fullPath, files);
            continue;
        }

        if (endsWithIgnoreCase(name, ".txt")) {
            files.push_back(fullPath);
        }
    }

    closedir(dir);
}

std::vector<std::string> listTxtFilesInDirectory(const std::string& directoryPath) {
    std::vector<std::string> files;
    collectTxtFilesRecursive(directoryPath, files);
    std::sort(files.begin(), files.end());
    return files;
}
#endif

bool isDirectoryPath(const std::string& path) {
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(path.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        return false;
    }
    return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
#endif
}

bool mergeTxtFilesToCorpus(
    const std::vector<std::string>& txtFiles,
    const std::string& outputCorpusPath) {
    const std::size_t pos = outputCorpusPath.find_last_of("/\\");
    if (pos != std::string::npos) {
        const std::string directory = outputCorpusPath.substr(0, pos);
#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        mkdir(directory.c_str(), 0755);
#endif
    }

    std::ofstream out(outputCorpusPath.c_str(), std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    std::vector<std::ifstream> inputs;
    inputs.reserve(txtFiles.size());
    for (std::size_t i = 0; i < txtFiles.size(); ++i) {
        inputs.emplace_back(txtFiles[i].c_str());
    }

    std::vector<bool> exhausted(inputs.size(), false);
    std::size_t remainingFiles = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].is_open()) {
            ++remainingFiles;
        } else {
            exhausted[i] = true;
        }
    }

    while (remainingFiles > 0) {
        bool wroteAnyLineInPass = false;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (exhausted[i]) {
                continue;
            }

            std::string line;
            if (std::getline(inputs[i], line)) {
                out << line << '\n';
                wroteAnyLineInPass = true;
                continue;
            }

            exhausted[i] = true;
            --remainingFiles;
        }

        if (!wroteAnyLineInPass) {
            break;
        }
    }

    return true;
}

bool resolveTrainingCorpusPath(
    const std::string& configuredPath,
    const std::string& mergedOutputPath,
    std::string& resolvedPath) {
    if (!isDirectoryPath(configuredPath)) {
        resolvedPath = configuredPath;
        return true;
    }

    const std::vector<std::string> txtFiles = listTxtFilesInDirectory(configuredPath);
    if (txtFiles.empty()) {
        return false;
    }

    if (!mergeTxtFilesToCorpus(txtFiles, mergedOutputPath)) {
        return false;
    }

    resolvedPath = mergedOutputPath;
    return true;
}

std::string formatPieceForTerminal(const std::string& piece) {
    const std::string marker = "\xE2\x96\x81";
    std::string out = piece;
    if (out == marker) {
        return "";
    }

    std::size_t pos = 0;
    while ((pos = out.find(marker, pos)) != std::string::npos) {
        out.erase(pos, marker.size());
    }
    return out;
}

std::vector<std::string> buildDisplayPieces(const std::vector<std::string>& pieces) {
    const std::string marker = "\xE2\x96\x81";
    bool hasSentencePieceMarker = false;
    for (const std::string& piece : pieces) {
        if (piece.find(marker) != std::string::npos) {
            hasSentencePieceMarker = true;
            break;
        }
    }

    if (!hasSentencePieceMarker) {
        return pieces;
    }

    std::vector<std::string> display;
    std::string current;

    auto flushCurrent = [&]() {
        if (!current.empty()) {
            display.push_back(current);
            current.clear();
        }
    };

    for (const std::string& piece : pieces) {
        if (piece == marker) {
            flushCurrent();
            continue;
        }

        const std::string cleaned = formatPieceForTerminal(piece);
        if (cleaned.empty()) {
            flushCurrent();
            continue;
        }

        if (piece.rfind(marker, 0) == 0) {
            flushCurrent();
        }
        current.append(cleaned);
    }

    flushCurrent();
    return display;
}

NKS_Tokenizer createBpeTokenizer() {
    const BpeRuntimeConfig cfg;
    NKS_Tokenizer tokenizer;
    NKS_Tokenizer::BpeTrainingConfig trainingCfg;
    trainingCfg.mergeOps = cfg.mergeOps;
    trainingCfg.trainingWordLimit = cfg.trainingWordLimit;
    trainingCfg.showProgress = cfg.showTrainingProgress;

    tokenizer
        .setLowercase(true)
        .setSplitOnPunctuation(true)
        .setKeepPunctuation(true)
        .setSplitCamelCase(true)
        .setTrainingConfig(trainingCfg)
        .setPreserveUnknownTokens(true);
    return tokenizer;
}

NKS_SentencePieceTokenizer createSentencePieceTokenizer() {
    const SentencePieceRuntimeConfig cfg;
    NKS_SentencePieceTokenizer tokenizer;
    NKS_SentencePieceTokenizer::TrainingConfig trainingCfg;
    trainingCfg.lowercase = true;
    trainingCfg.splitCamelCase = true;
    trainingCfg.targetVocabSize = cfg.targetVocabSize;
    trainingCfg.maxPieceChars = cfg.maxPieceChars;
    trainingCfg.trainingLineLimit = cfg.trainingLineLimit;

    tokenizer.setTrainingConfig(trainingCfg);
    return tokenizer;
}

bool loadOrTrainBpeModelOrReport(
    NKS_Tokenizer& tokenizer,
    const std::string& vocabularyPath,
    const std::string& modelPath,
    const std::string& mergedCorpusPath) {
    if (tokenizer.loadModel(modelPath)) {
        std::cout << "Loaded BPE model from metadata: " << modelPath << std::endl;
        return true;
    }

    std::string resolvedTrainingPath;
    if (!resolveTrainingCorpusPath(vocabularyPath, mergedCorpusPath, resolvedTrainingPath)) {
        std::cerr << "Failed to resolve BPE training corpus from path: " << vocabularyPath << std::endl;
        return false;
    }

    std::cout << "Metadata model not found/invalid. Training BPE model..." << std::endl;
    if (!tokenizer.loadVocabulary(resolvedTrainingPath)) {
        std::cerr << "Failed to train BPE model from vocabulary file: " << resolvedTrainingPath << std::endl;
        return false;
    }
    printBpeTrainingStats(tokenizer.lastTrainingStats());

    const std::size_t pos = modelPath.find_last_of("/\\");
    if (pos != std::string::npos) {
        const std::string directory = modelPath.substr(0, pos);
#ifdef _WIN32
        _mkdir(directory.c_str());
#else
        mkdir(directory.c_str(), 0755);
#endif
    }

    if (tokenizer.saveModel(modelPath)) {
        std::cout << "Saved BPE model metadata: " << modelPath << std::endl;
    } else {
        std::cerr << "Warning: failed to save BPE model metadata to " << modelPath << std::endl;
    }

    return true;
}

bool trainSentencePieceOrReport(NKS_SentencePieceTokenizer& tokenizer, const std::string& corpusPath) {
    if (tokenizer.trainFromFile(corpusPath)) {
        return true;
    }
    std::cerr << "Failed to train SentencePiece from file: " << corpusPath << std::endl;
    return false;
}

TokenizationResult runBpePipeline(NKS_Tokenizer& tokenizer, const std::string& text) {
    TokenizationResult result;
    result.pieces = tokenizer.tokenize(text);
    result.tokenIds = tokenizer.encode(text);
    result.decodedText = tokenizer.decode(result.tokenIds);
    result.approxModelTokenCount = tokenizer.estimateModelTokensApprox(text);
    result.vocabularySize = tokenizer.vocabularySize();
    return result;
}

TokenizationResult runSentencePiecePipeline(const NKS_SentencePieceTokenizer& tokenizer, const std::string& text) {
    TokenizationResult result;
    result.pieces = tokenizer.encode(text);
    result.tokenIds = tokenizer.encodeToIds(text);
    result.decodedText = tokenizer.decode(result.pieces);
    result.approxModelTokenCount =
        static_cast<std::size_t>(std::ceil(static_cast<double>(text.size()) / kApproxCharsPerToken));
    result.vocabularySize = tokenizer.vocabularySize();
    return result;
}

void printPieces(const std::vector<std::string>& pieces) {
    const std::vector<std::string> displayPieces = buildDisplayPieces(pieces);
    std::cout << "Tokenizer pieces: ";
    for (std::size_t i = 0; i < displayPieces.size(); ++i) {
        std::cout << "[" << displayPieces[i] << "]";
        if (i + 1 < displayPieces.size()) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

void printTokenIds(const std::vector<int>& tokenIds) {
    std::cout << "Encoded token IDs: ";
    for (std::size_t i = 0; i < tokenIds.size(); ++i) {
        std::cout << tokenIds[i];
        if (i + 1 < tokenIds.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

void printSummary(const std::string& mode, const std::string& inputText, const TokenizationResult& result) {
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Vocabulary size: " << result.vocabularySize << std::endl;
    std::cout << "Input text: " << inputText << std::endl;
    std::cout << "Approx model token count: " << result.approxModelTokenCount << std::endl;
    printPieces(result.pieces);
    printTokenIds(result.tokenIds);
    std::cout << "Decoded text: " << result.decodedText << std::endl;
}

TokenizerMode parseTokenizerMode() {
    std::cout << "Choose tokenizer mode [bpe/sentencepiece] (default=bpe): ";
    std::string mode;
    std::getline(std::cin, mode);
    if (mode.empty()) {
        return TokenizerMode::kBpe;
    }

    for (char& c : mode) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (mode == "sp" || mode == "sentencepiece") {
        return TokenizerMode::kSentencePiece;
    }
    if (mode == "bpe") {
        return TokenizerMode::kBpe;
    }
    return TokenizerMode::kUnsupported;
}

std::string readInputTextFromTerminal() {
    std::cout << "Enter text to tokenize: ";
    std::string inputText;
    std::getline(std::cin, inputText);
    return inputText;
}
} // namespace

int runTokenizerApplication() {
    const AppPaths paths;

    const TokenizerMode mode = parseTokenizerMode();
    const std::string inputText = readInputTextFromTerminal();
    if (inputText.empty()) {
        std::cerr << "No input provided." << std::endl;
        return 1;
    }

    if (mode == TokenizerMode::kBpe) {
        NKS_Tokenizer tokenizer = createBpeTokenizer();
        if (!loadOrTrainBpeModelOrReport(
                tokenizer,
                paths.bpeTrainingPath,
                paths.bpeModelPath,
                paths.mergedTxtCorpusPath)) {
            return 1;
        }

        const TokenizationResult result = runBpePipeline(tokenizer, inputText);
        printSummary("bpe", inputText, result);
        return 0;
    }

    if (mode == TokenizerMode::kSentencePiece) {
        std::string resolvedTrainingPath;
        if (!resolveTrainingCorpusPath(
                paths.sentencePieceTrainingPath,
                paths.mergedTxtCorpusPath,
                resolvedTrainingPath)) {
            std::cerr << "Failed to resolve SentencePiece training corpus from path: "
                      << paths.sentencePieceTrainingPath << std::endl;
            return 1;
        }

        NKS_SentencePieceTokenizer tokenizer = createSentencePieceTokenizer();
        if (!trainSentencePieceOrReport(tokenizer, resolvedTrainingPath)) {
            return 1;
        }

        const TokenizationResult result = runSentencePiecePipeline(tokenizer, inputText);
        printSummary("sentencepiece", inputText, result);
        return 0;
    }

    std::cerr << "Unsupported mode. Use 'bpe' or 'sentencepiece'." << std::endl;
    return 1;
}

int runCompressionExample() {
    try {
        clm::CompressorConfig cfg{
            512U * 1024U,                      // chunk_size
            0U,                                // num_threads (auto)
            true,                              // lazy_match
            false,                             // verbose
            [](uint32_t, uint32_t) {}          // progress callback
        };
        clm::Compressor comp(cfg);
        // comp.compress_file("Data/Training_Essay_Data.txt", "Data/Training_Essay_Data.clm");
        comp.compress_file("Data/exampleText.txt", "Data/exampleText.clm");
        std::cout << "Compression completed: Data/exampleText.clm" << std::endl;
        comp.decompress_file("Data/exampleText.clm", "Data/exampleText_decompressed.txt");
        std::cout << "Decompression completed: Data/exampleText_decompressed.txt" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Compression failed: " << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Compression failed: unknown error" << std::endl;
        return 1;
    }
}

int runLLMExample() {
    using namespace nks_llm;

    std::cout << "\n========================================" << std::endl;
    std::cout << "    LLM from Scratch - Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create a small model for demonstration (faster initialization)
    std::cout << "[1] Creating LLM model..." << std::endl;
    ModelConfig config = ModelConfig::get_small_model();
    config.batch_size = 1;
    config.max_seq_length = 128;
    
    std::cout << "  - Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  - Embedding dim: " << config.embedding_dim << std::endl;
    std::cout << "  - Num layers: " << config.num_layers << std::endl;
    std::cout << "  - Num heads: " << config.num_heads << std::endl;
    
    LLMModel model(config);
    std::cout << "  - Total parameters: " << model.num_parameters() / 1e6 << "M" << std::endl;

    // Create sample input
    std::cout << "\n[2] Creating sample input..." << std::endl;
    std::vector<int> sample_input = {1, 5, 10, 15, 20, 25, 30};
    while (sample_input.size() < config.max_seq_length / 2) {
        sample_input.push_back(rand() % config.vocab_size);
    }
    
    Tensor input_ids({1, static_cast<size_t>(sample_input.size())});
    for (size_t i = 0; i < sample_input.size(); ++i) {
        input_ids[i] = static_cast<float>(sample_input[i]);
    }
    std::cout << "  - Input shape: " << input_ids.shape()[0] << " x " << input_ids.shape()[1] << std::endl;

    // Forward pass
    std::cout << "\n[3] Running forward pass..." << std::endl;
    try {
        Tensor logits = model.forward(input_ids);
        std::cout << "  - Logits shape: " << logits.shape()[0] << " x " 
                  << logits.shape()[1] << " x " << logits.shape()[2] << std::endl;
        std::cout << "  ✓ Forward pass successful!" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Forward pass failed: " << ex.what() << std::endl;
        return 1;
    }

    // Training step example
    std::cout << "\n[4] Simulating training step..." << std::endl;
    try {
        Tensor target_ids({1, static_cast<size_t>(sample_input.size())});
        for (size_t i = 0; i < sample_input.size(); ++i) {
            target_ids[i] = static_cast<float>((sample_input[i] + 1) % config.vocab_size);
        }
        
        LLMModel::TrainStep step = model.training_step(input_ids, target_ids);
        std::cout << "  - Loss: " << step.loss << std::endl;
        std::cout << "  - Perplexity: " << step.perplexity << std::endl;
        std::cout << "  ✓ Training step successful!" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Training step failed: " << ex.what() << std::endl;
        return 1;
    }

    // Generation example
    std::cout << "\n[5] Generating tokens..." << std::endl;
    try {
        std::vector<int> prompt = {1, 5, 10};
        auto generated = model.generate(prompt, 10);
        std::cout << "  - Prompt: [1, 5, 10]" << std::endl;
        std::cout << "  - Generated: [";
        for (size_t i = 3; i < generated.size(); ++i) {
            if (i > 3) std::cout << ", ";
            std::cout << generated[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  ✓ Generation successful!" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Generation failed: " << ex.what() << std::endl;
        return 1;
    }

    // Save/Load checkpoint
    std::cout << "\n[6] Testing checkpoint save/load..." << std::endl;
    try {
        std::string checkpoint_path = "Metadata/llm_checkpoint.bin";
        if (model.save(checkpoint_path)) {
            std::cout << "  ✓ Model saved to: " << checkpoint_path << std::endl;
        } else {
            std::cerr << "  ✗ Failed to save model" << std::endl;
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Checkpoint failed: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "    Demo completed successfully!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
