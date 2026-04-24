#include "app_runner.h"

#include <cmath>
#include <cctype>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
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
    std::string chatModelPath = getEnvOrDefault("NKS_CHAT_MODEL_PATH", "Metadata/llm_chat_ngram.bin");
};

struct TokenPairKey {
    int first = -1;
    int second = -1;

    bool operator==(const TokenPairKey& other) const {
        return first == other.first && second == other.second;
    }
};

struct TokenPairKeyHash {
    std::size_t operator()(const TokenPairKey& key) const {
        const std::uint64_t a = static_cast<std::uint32_t>(key.first);
        const std::uint64_t b = static_cast<std::uint32_t>(key.second);
        return static_cast<std::size_t>((a << 32U) ^ b);
    }
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
constexpr std::size_t kDefaultChatTrainingLineLimit = 50000;
constexpr std::size_t kChatGenerationTokens = 48;

struct TokenizationResult {
    std::vector<std::string> pieces;
    std::vector<int> tokenIds;
    std::string decodedText;
    std::size_t approxModelTokenCount = 0;
    std::size_t vocabularySize = 0;
};

struct ChatNgramModel {
    std::unordered_map<int, std::unordered_map<int, std::uint32_t>> transitions;
    std::unordered_map<TokenPairKey, std::unordered_map<int, std::uint32_t>, TokenPairKeyHash> pairTransitions;
    std::unordered_map<int, std::uint32_t> unigramCounts;
    std::size_t vocabSize = 0;
    std::size_t observedPairs = 0;
    std::size_t observedTriples = 0;

    void clear() {
        transitions.clear();
        pairTransitions.clear();
        unigramCounts.clear();
        vocabSize = 0;
        observedPairs = 0;
        observedTriples = 0;
    }

    void observe(const std::vector<int>& tokenIds, std::size_t tokenizerVocabSize) {
        if (tokenIds.empty()) {
            return;
        }

        vocabSize = std::max(vocabSize, tokenizerVocabSize);
        int previousPrevious = -1;
        int previous = -1;
        for (int tokenId : tokenIds) {
            if (tokenId < 0 || tokenId >= static_cast<int>(tokenizerVocabSize)) {
                previous = -1;
                continue;
            }

            ++unigramCounts[tokenId];
            if (previous >= 0) {
                ++transitions[previous][tokenId];
                ++observedPairs;
            }
            if (previousPrevious >= 0 && previous >= 0) {
                ++pairTransitions[TokenPairKey{previousPrevious, previous}][tokenId];
                ++observedTriples;
            }
            previousPrevious = previous;
            previous = tokenId;
        }
    }

    bool empty() const {
        return observedPairs == 0 || unigramCounts.empty();
    }

    int sampleFromCounts(
        const std::unordered_map<int, std::uint32_t>& counts,
        std::mt19937& gen,
        const std::deque<int>& recent) const {
        if (counts.empty()) {
            return 0;
        }

        std::vector<int> ids;
        std::vector<double> weights;
        ids.reserve(counts.size());
        weights.reserve(counts.size());

        for (const auto& kv : counts) {
            double weight = static_cast<double>(kv.second);
            for (int recentToken : recent) {
                if (recentToken == kv.first) {
                    weight *= 0.35;
                }
            }
            ids.push_back(kv.first);
            weights.push_back(std::max(weight, 0.001));
        }

        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        return ids[dist(gen)];
    }

    std::vector<int> generate(const std::vector<int>& promptIds, std::size_t maxNewTokens) const {
        if (empty()) {
            return {};
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> generated;
        generated.reserve(maxNewTokens);

        int previous = -1;
        int current = -1;
        for (auto it = promptIds.rbegin(); it != promptIds.rend(); ++it) {
            if (*it >= 0 && *it < static_cast<int>(vocabSize)) {
                if (current < 0) {
                    current = *it;
                } else {
                    previous = *it;
                    break;
                }
            }
        }

        std::deque<int> recent;
        if (current >= 0) {
            recent.push_back(current);
        }

        for (std::size_t i = 0; i < maxNewTokens; ++i) {
            int nextToken = 0;
            const auto pairIt = pairTransitions.find(TokenPairKey{previous, current});
            if (pairIt != pairTransitions.end() && !pairIt->second.empty()) {
                nextToken = sampleFromCounts(pairIt->second, gen, recent);
            } else {
                const auto transitionIt = transitions.find(current);
                if (transitionIt != transitions.end() && !transitionIt->second.empty()) {
                    nextToken = sampleFromCounts(transitionIt->second, gen, recent);
                } else {
                    nextToken = sampleFromCounts(unigramCounts, gen, recent);
                }
            }

            generated.push_back(nextToken);
            previous = current;
            current = nextToken;
            recent.push_back(nextToken);
            if (recent.size() > 24) {
                recent.pop_front();
            }
        }

        return generated;
    }

    bool save(const std::string& path) const {
        const std::size_t pos = path.find_last_of("/\\");
        if (pos != std::string::npos) {
            const std::string directory = path.substr(0, pos);
#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            mkdir(directory.c_str(), 0755);
#endif
        }

        std::ofstream out(path.c_str(), std::ios::binary);
        if (!out.is_open()) {
            return false;
        }

        const char magic[8] = {'N', 'K', 'S', 'N', 'G', 'R', 'M', '2'};
        out.write(magic, sizeof(magic));

        const std::uint64_t savedVocabSize = static_cast<std::uint64_t>(vocabSize);
        const std::uint64_t savedObservedPairs = static_cast<std::uint64_t>(observedPairs);
        const std::uint64_t savedObservedTriples = static_cast<std::uint64_t>(observedTriples);
        const std::uint64_t unigramSize = static_cast<std::uint64_t>(unigramCounts.size());
        const std::uint64_t transitionSize = static_cast<std::uint64_t>(transitions.size());
        const std::uint64_t pairTransitionSize = static_cast<std::uint64_t>(pairTransitions.size());
        out.write(reinterpret_cast<const char*>(&savedVocabSize), sizeof(savedVocabSize));
        out.write(reinterpret_cast<const char*>(&savedObservedPairs), sizeof(savedObservedPairs));
        out.write(reinterpret_cast<const char*>(&savedObservedTriples), sizeof(savedObservedTriples));
        out.write(reinterpret_cast<const char*>(&unigramSize), sizeof(unigramSize));
        out.write(reinterpret_cast<const char*>(&transitionSize), sizeof(transitionSize));
        out.write(reinterpret_cast<const char*>(&pairTransitionSize), sizeof(pairTransitionSize));

        for (const auto& kv : unigramCounts) {
            const std::int32_t id = static_cast<std::int32_t>(kv.first);
            const std::uint32_t count = kv.second;
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        }

        for (const auto& row : transitions) {
            const std::int32_t prev = static_cast<std::int32_t>(row.first);
            const std::uint64_t rowSize = static_cast<std::uint64_t>(row.second.size());
            out.write(reinterpret_cast<const char*>(&prev), sizeof(prev));
            out.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));
            for (const auto& kv : row.second) {
                const std::int32_t next = static_cast<std::int32_t>(kv.first);
                const std::uint32_t count = kv.second;
                out.write(reinterpret_cast<const char*>(&next), sizeof(next));
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }

        for (const auto& row : pairTransitions) {
            const std::int32_t first = static_cast<std::int32_t>(row.first.first);
            const std::int32_t second = static_cast<std::int32_t>(row.first.second);
            const std::uint64_t rowSize = static_cast<std::uint64_t>(row.second.size());
            out.write(reinterpret_cast<const char*>(&first), sizeof(first));
            out.write(reinterpret_cast<const char*>(&second), sizeof(second));
            out.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));
            for (const auto& kv : row.second) {
                const std::int32_t next = static_cast<std::int32_t>(kv.first);
                const std::uint32_t count = kv.second;
                out.write(reinterpret_cast<const char*>(&next), sizeof(next));
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }

        return out.good();
    }

    bool load(const std::string& path) {
        clear();
        std::ifstream in(path.c_str(), std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        char magic[8] = {};
        in.read(magic, sizeof(magic));
        const char expected[8] = {'N', 'K', 'S', 'N', 'G', 'R', 'M', '2'};
        if (!std::equal(magic, magic + sizeof(magic), expected)) {
            return false;
        }

        std::uint64_t savedVocabSize = 0;
        std::uint64_t savedObservedPairs = 0;
        std::uint64_t savedObservedTriples = 0;
        std::uint64_t unigramSize = 0;
        std::uint64_t transitionSize = 0;
        std::uint64_t pairTransitionSize = 0;
        in.read(reinterpret_cast<char*>(&savedVocabSize), sizeof(savedVocabSize));
        in.read(reinterpret_cast<char*>(&savedObservedPairs), sizeof(savedObservedPairs));
        in.read(reinterpret_cast<char*>(&savedObservedTriples), sizeof(savedObservedTriples));
        in.read(reinterpret_cast<char*>(&unigramSize), sizeof(unigramSize));
        in.read(reinterpret_cast<char*>(&transitionSize), sizeof(transitionSize));
        in.read(reinterpret_cast<char*>(&pairTransitionSize), sizeof(pairTransitionSize));

        vocabSize = static_cast<std::size_t>(savedVocabSize);
        observedPairs = static_cast<std::size_t>(savedObservedPairs);
        observedTriples = static_cast<std::size_t>(savedObservedTriples);

        for (std::uint64_t i = 0; i < unigramSize; ++i) {
            std::int32_t id = 0;
            std::uint32_t count = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            unigramCounts[static_cast<int>(id)] = count;
        }

        for (std::uint64_t i = 0; i < transitionSize; ++i) {
            std::int32_t prev = 0;
            std::uint64_t rowSize = 0;
            in.read(reinterpret_cast<char*>(&prev), sizeof(prev));
            in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
            auto& row = transitions[static_cast<int>(prev)];
            for (std::uint64_t j = 0; j < rowSize; ++j) {
                std::int32_t next = 0;
                std::uint32_t count = 0;
                in.read(reinterpret_cast<char*>(&next), sizeof(next));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                row[static_cast<int>(next)] = count;
            }
        }

        for (std::uint64_t i = 0; i < pairTransitionSize; ++i) {
            std::int32_t first = 0;
            std::int32_t second = 0;
            std::uint64_t rowSize = 0;
            in.read(reinterpret_cast<char*>(&first), sizeof(first));
            in.read(reinterpret_cast<char*>(&second), sizeof(second));
            in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
            auto& row = pairTransitions[TokenPairKey{static_cast<int>(first), static_cast<int>(second)}];
            for (std::uint64_t j = 0; j < rowSize; ++j) {
                std::int32_t next = 0;
                std::uint32_t count = 0;
                in.read(reinterpret_cast<char*>(&next), sizeof(next));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                row[static_cast<int>(next)] = count;
            }
        }

        return in.good();
    }
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

std::vector<int> clampTokenIdsToModelVocab(const std::vector<int>& tokenIds, std::size_t vocabSize) {
    std::vector<int> clamped;
    clamped.reserve(tokenIds.size());

    const int safeVocab = static_cast<int>(std::max<std::size_t>(vocabSize, 1));
    for (int tokenId : tokenIds) {
        if (tokenId < 0) {
            clamped.push_back(0);
            continue;
        }
        clamped.push_back(tokenId % safeVocab);
    }

    return clamped;
}

void printGeneratedTokenIds(const std::vector<int>& tokenIds) {
    std::cout << "LLM token IDs: [";
    for (std::size_t i = 0; i < tokenIds.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << tokenIds[i];
    }
    std::cout << "]" << std::endl;
}

std::size_t readSizeFromTerminal(const std::string& prompt, std::size_t fallback) {
    std::cout << prompt << " (default=" << fallback << "): ";
    std::string value;
    std::getline(std::cin, value);
    if (value.empty()) {
        return fallback;
    }

    try {
        const std::size_t parsed = static_cast<std::size_t>(std::stoull(value));
        return parsed == 0 ? fallback : parsed;
    } catch (...) {
        std::cerr << "Invalid number. Using default: " << fallback << std::endl;
        return fallback;
    }
}

bool trainChatModelFromCorpus(
    ChatNgramModel& chatModel,
    NKS_Tokenizer& tokenizer,
    const std::string& corpusPath,
    std::size_t epochs,
    std::size_t lineLimit) {
    const std::size_t tokenizerVocabSize = tokenizer.vocabularySize();
    if (tokenizerVocabSize == 0) {
        std::cerr << "Tokenizer vocabulary is empty." << std::endl;
        return false;
    }

    chatModel.clear();
    chatModel.vocabSize = tokenizerVocabSize;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::ifstream in(corpusPath.c_str());
        if (!in.is_open()) {
            std::cerr << "Failed to open training corpus: " << corpusPath << std::endl;
            return false;
        }

        std::string line;
        std::size_t linesRead = 0;
        std::size_t tokensSeen = 0;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }

            std::vector<int> tokenIds = tokenizer.encode(line);
            tokensSeen += tokenIds.size();
            chatModel.observe(tokenIds, tokenizerVocabSize);

            ++linesRead;
            if (linesRead % 1000 == 0) {
                std::cout << "\r  Epoch " << (epoch + 1) << "/" << epochs
                          << " | lines=" << linesRead
                          << " | tokens=" << tokensSeen
                          << " | pairs=" << chatModel.observedPairs << std::flush;
            }

            if (lineLimit > 0 && linesRead >= lineLimit) {
                break;
            }
        }

        std::cout << "\r  Epoch " << (epoch + 1) << "/" << epochs
                  << " completed | lines=" << linesRead
                  << " | tokens=" << tokensSeen
                  << " | pairs=" << chatModel.observedPairs
                  << " | triples=" << chatModel.observedTriples
                  << "                    " << std::endl;
    }

    return !chatModel.empty();
}

void printChatModelSample(ChatNgramModel& chatModel, NKS_Tokenizer& tokenizer, const std::string& prompt) {
    const std::vector<int> promptIds = tokenizer.encode(prompt);
    const std::vector<int> sampleIds = chatModel.generate(promptIds, kChatGenerationTokens);
    std::cout << "\nPrompt> " << prompt << std::endl;
    std::cout << "Model> " << tokenizer.decode(sampleIds) << std::endl;
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
    std::cout << "  - Compute backend: " << gpu_backend::backend_name();
    if (!gpu_backend::is_available()) {
        std::cout << " (GPU unavailable or CUDA support not compiled)";
    }
    std::cout << std::endl;
    
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

int runLLMTrainingExample() {
    using namespace nks_llm;

    std::cout << "\n========================================" << std::endl;
    std::cout << "   LLM Training Loop - Full Example" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // [1] Create model
    std::cout << "[1] Creating LLM model for training..." << std::endl;
    ModelConfig config = ModelConfig::get_small_model();
    config.batch_size = 4;
    config.max_seq_length = 64;
    config.num_epochs = 3;
    config.learning_rate = 1e-3f;
    config.weight_decay = 1e-5f;
    config.gradient_clip = 1.0f;
    
    std::cout << "  - Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  - Embedding dim: " << config.embedding_dim << std::endl;
    std::cout << "  - Num layers: " << config.num_layers << std::endl;
    std::cout << "  - Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  - Weight decay: " << config.weight_decay << std::endl;
    std::cout << "  - Compute backend: " << gpu_backend::backend_name();
    if (!gpu_backend::is_available()) {
        std::cout << " (GPU unavailable or CUDA support not compiled)";
    }
    std::cout << std::endl;
    
    LLMModel model(config);
    std::cout << "  - Total parameters: " << model.num_parameters() / 1e6 << "M" << std::endl;

    // [2] Create synthetic training data
    std::cout << "\n[2] Creating synthetic training batches..." << std::endl;
    
    size_t num_batches = 5;
    std::vector<Tensor> input_batches;
    std::vector<Tensor> target_batches;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(config.vocab_size - 1));
    
    for (size_t b = 0; b < num_batches; ++b) {
        // Input batch: (batch_size, seq_length)
        Tensor input_batch({config.batch_size, 32});
        Tensor target_batch({config.batch_size, 32});
        
        for (size_t i = 0; i < input_batch.elem_count(); ++i) {
            int token_id = dis(gen);
            input_batch[i] = static_cast<float>(token_id);
            target_batch[i] = static_cast<float>((token_id + 1) % config.vocab_size);
        }
        
        input_batches.push_back(input_batch);
        target_batches.push_back(target_batch);
    }
    
    std::cout << "  - Created " << num_batches << " batches" << std::endl;
    std::cout << "  - Batch size: " << config.batch_size << std::endl;
    std::cout << "  - Sequence length: 32 tokens" << std::endl;

    // [3] Training loop with epoch management
    std::cout << "\n[3] Starting training loop..." << std::endl;
    std::cout << "  - Epochs: " << config.num_epochs << std::endl;
    std::cout << "  - Batches per epoch: " << num_batches << std::endl;
    std::cout << "  - Optimizer: Adam" << std::endl;
    std::cout << "  - LR Schedule: Cosine Annealing\n" << std::endl;
    
    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::cout << "╔════════════════════════════════════════╗" << std::endl;
        std::cout << "  Epoch " << epoch + 1 << "/" << config.num_epochs << std::endl;
        std::cout << "╚════════════════════════════════════════╝" << std::endl;
        
        try {
            auto stats = model.train_epoch(input_batches, target_batches, epoch, config.num_epochs);
            
            std::cout << "\n  ✓ Epoch completed:" << std::endl;
            std::cout << "    - Final loss: " << stats.loss << std::endl;
            std::cout << "    - Avg loss: " << stats.avg_loss << std::endl;
            std::cout << "    - Perplexity: " << stats.perplexity << std::endl;
            std::cout << "    - Gradient norm: " << stats.gradient_norm << std::endl;
            std::cout << "    - Learning rate: " << std::scientific << stats.learning_rate 
                      << std::defaultfloat << std::endl;
            
        } catch (const std::exception& ex) {
            std::cerr << "  ✗ Training failed: " << ex.what() << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
    }

    // [4] Validation on test data
    std::cout << "[4] Running validation..." << std::endl;
    try {
        Tensor val_input({1, 32});
        Tensor val_target({1, 32});
        
        for (size_t i = 0; i < val_input.elem_count(); ++i) {
            int token_id = dis(gen);
            val_input[i] = static_cast<float>(token_id);
            val_target[i] = static_cast<float>((token_id + 1) % config.vocab_size);
        }
        
        auto step = model.training_step(val_input, val_target);
        
        std::cout << "  - Validation loss: " << step.loss << std::endl;
        std::cout << "  - Validation perplexity: " << step.perplexity << std::endl;
        std::cout << "  ✓ Validation completed!" << std::endl;
        
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Validation failed: " << ex.what() << std::endl;
        return 1;
    }

    // [5] Test generation with trained model
    std::cout << "\n[5] Testing generation with trained model..." << std::endl;
    try {
        std::vector<int> prompt = {1, 5, 10, 15};
        auto generated = model.generate(prompt, 16);
        
        std::cout << "  - Prompt: [1, 5, 10, 15]" << std::endl;
        std::cout << "  - Generated: [";
        for (size_t i = 4; i < generated.size(); ++i) {
            if (i > 4) std::cout << ", ";
            std::cout << generated[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  ✓ Generation successful!" << std::endl;
        
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Generation failed: " << ex.what() << std::endl;
        return 1;
    }

    // [6] Save trained checkpoint
    std::cout << "\n[6] Saving trained checkpoint..." << std::endl;
    try {
        std::string checkpoint_path = "Metadata/llm_trained_checkpoint.bin";
        if (model.save(checkpoint_path)) {
            std::cout << "  ✓ Trained model saved to: " << checkpoint_path << std::endl;
        } else {
            std::cerr << "  ✗ Failed to save checkpoint" << std::endl;
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "  ✗ Checkpoint save failed: " << ex.what() << std::endl;
        return 1;
    }

    // [7] Print training summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Training Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "✓ Model initialized with " << model.num_parameters() / 1e6 << "M parameters" << std::endl;
    std::cout << "✓ Trained for " << config.num_epochs << " epochs" << std::endl;
    std::cout << "✓ Total training steps: " << config.num_epochs * num_batches << std::endl;
    std::cout << "✓ Final model checkpoint saved" << std::endl;
    std::cout << "✓ Model ready for generation and fine-tuning" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}

int runRealCorpusTrainingExample() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Real Corpus Next-Token Training" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const AppPaths paths;

    std::cout << "[1] Loading tokenizer..." << std::endl;
    NKS_Tokenizer tokenizer = createBpeTokenizer();
    if (!loadOrTrainBpeModelOrReport(
            tokenizer,
            paths.bpeTrainingPath,
            paths.bpeModelPath,
            paths.mergedTxtCorpusPath)) {
        return 1;
    }

    std::string corpusPath;
    if (!resolveTrainingCorpusPath(paths.bpeTrainingPath, paths.mergedTxtCorpusPath, corpusPath)) {
        std::cerr << "Failed to resolve training corpus from: " << paths.bpeTrainingPath << std::endl;
        return 1;
    }

    std::cout << "  - Tokenizer vocabulary size: " << tokenizer.vocabularySize() << std::endl;
    std::cout << "  - Training corpus: " << corpusPath << std::endl;
    std::cout << "\nThis trains a real next-token model from corpus token IDs and saves it for chat." << std::endl;

    const std::size_t epochs = readSizeFromTerminal("\nEnter epochs", 10);
    const std::size_t lineLimit = readSizeFromTerminal(
        "Enter max lines per epoch, 0 for full corpus",
        kDefaultChatTrainingLineLimit);

    std::cout << "\n[2] Training..." << std::endl;
    ChatNgramModel chatModel;
    if (!trainChatModelFromCorpus(chatModel, tokenizer, corpusPath, epochs, lineLimit)) {
        std::cerr << "Training failed or produced no token pairs." << std::endl;
        return 1;
    }

    std::cout << "\n[3] Saving trained chat model..." << std::endl;
    if (!chatModel.save(paths.chatModelPath)) {
        std::cerr << "Failed to save chat model: " << paths.chatModelPath << std::endl;
        return 1;
    }

    const std::vector<int> testPrompt = tokenizer.encode("how are you");
    const std::vector<int> sampleIds = chatModel.generate(testPrompt, 24);
    std::cout << "  - Saved: " << paths.chatModelPath << std::endl;
    std::cout << "  - Sample prompt: how are you" << std::endl;
    std::cout << "  - Sample output: " << tokenizer.decode(sampleIds) << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "   Training Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Trained for " << epochs << " epochs" << std::endl;
    std::cout << "Observed token pairs: " << chatModel.observedPairs << std::endl;
    std::cout << "Observed token triples: " << chatModel.observedTriples << std::endl;
    std::cout << "Transition rows: " << chatModel.transitions.size() << std::endl;
    std::cout << "Pair transition rows: " << chatModel.pairTransitions.size() << std::endl;
    std::cout << "Saved chat model: " << paths.chatModelPath << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}

int runLLMChatExample() {
    using namespace nks_llm;

    const AppPaths paths;

    std::cout << "\n========================================" << std::endl;
    std::cout << "        LLM Terminal Chat" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "[1] Loading tokenizer..." << std::endl;
    NKS_Tokenizer tokenizer = createBpeTokenizer();
    if (!loadOrTrainBpeModelOrReport(
            tokenizer,
            paths.bpeTrainingPath,
            paths.bpeModelPath,
            paths.mergedTxtCorpusPath)) {
        return 1;
    }
    std::cout << "  - Tokenizer vocabulary size: " << tokenizer.vocabularySize() << std::endl;

    std::cout << "\n[2] Creating model..." << std::endl;
    ModelConfig config = ModelConfig::get_small_model();
    config.batch_size = 1;
    config.max_seq_length = 128;
    config.temperature = 1.2f;
    config.top_k = 40;

    std::cout << "  - Model vocabulary size: " << config.vocab_size << std::endl;
    std::cout << "  - Max sequence length: " << config.max_seq_length << std::endl;
    std::cout << "  - Compute backend: " << gpu_backend::backend_name();
    if (!gpu_backend::is_available()) {
        std::cout << " (GPU unavailable or CUDA support not compiled)";
    }
    std::cout << std::endl;

    LLMModel model(config);

    const std::string trainedCheckpointPath = "Metadata/llm_trained_checkpoint.bin";
    const std::string demoCheckpointPath = "Metadata/llm_checkpoint.bin";
    std::ifstream trainedCheckpoint(trainedCheckpointPath.c_str(), std::ios::binary);
    std::ifstream demoCheckpoint(demoCheckpointPath.c_str(), std::ios::binary);

    if (trainedCheckpoint.good() && model.load(trainedCheckpointPath)) {
        std::cout << "  - Loaded checkpoint: " << trainedCheckpointPath << std::endl;
    } else if (demoCheckpoint.good() && model.load(demoCheckpointPath)) {
        std::cout << "  - Loaded checkpoint: " << demoCheckpointPath << std::endl;
    } else {
        std::cout << "  - No checkpoint found. Using randomly initialized weights." << std::endl;
    }

    ChatNgramModel chatModel;
    const bool hasRealChatModel = chatModel.load(paths.chatModelPath) && !chatModel.empty();
    if (hasRealChatModel) {
        std::cout << "  - Loaded real corpus chat model: " << paths.chatModelPath << std::endl;
        std::cout << "  - Chat token pairs: " << chatModel.observedPairs << std::endl;
        std::cout << "  - Chat token triples: " << chatModel.observedTriples << std::endl;
    } else {
        std::cout << "  - No real corpus chat model found. Run option 4 to train one." << std::endl;
    }

    std::cout << "\nType your message and press Enter. Press Ctrl+C to stop." << std::endl;
    if (hasRealChatModel) {
        std::cout << "Using real corpus next-token model for chat.\n" << std::endl;
    } else {
        std::cout << "Note: falling back to the transformer prototype, so responses may look rough.\n"
                  << std::endl;
    }

    std::string line;
    while (true) {
        std::cout << "You> ";
        if (!std::getline(std::cin, line)) {
            std::cout << std::endl;
            break;
        }

        if (line.empty()) {
            continue;
        }

        std::vector<int> promptIds = tokenizer.encode(line);
        if (promptIds.empty()) {
            std::cout << "Model> [no tokens produced]\n" << std::endl;
            continue;
        }

        if (hasRealChatModel) {
            try {
                const std::vector<int> newTokenIds = chatModel.generate(promptIds, kChatGenerationTokens);
                const std::string response = tokenizer.decode(newTokenIds);
                if (response.empty()) {
                    std::cout << "Model> [empty decoded response]" << std::endl;
                } else {
                    std::cout << "Model> " << response << std::endl;
                }
                printGeneratedTokenIds(newTokenIds);
                std::cout << std::endl;
            } catch (const std::exception& ex) {
                std::cerr << "Model error: " << ex.what() << "\n" << std::endl;
            }
            continue;
        }

        std::vector<int> modelPromptIds = clampTokenIdsToModelVocab(promptIds, config.vocab_size);
        if (modelPromptIds.size() > config.max_seq_length) {
            modelPromptIds.erase(
                modelPromptIds.begin(),
                modelPromptIds.end() - static_cast<std::ptrdiff_t>(config.max_seq_length));
        }

        try {
            const std::size_t maxNewTokens = 32;
            std::vector<int> generated = model.generate(modelPromptIds, maxNewTokens);
            std::vector<int> newTokenIds;
            if (generated.size() > modelPromptIds.size()) {
                newTokenIds.assign(generated.begin() + static_cast<std::ptrdiff_t>(modelPromptIds.size()),
                                   generated.end());
            }

            std::string response = tokenizer.decode(newTokenIds);
            if (response.empty()) {
                std::cout << "Model> [empty decoded response]" << std::endl;
            } else {
                std::cout << "Model> " << response << std::endl;
            }
            printGeneratedTokenIds(newTokenIds);
            std::cout << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "Model error: " << ex.what() << "\n" << std::endl;
        }
    }

    std::cout << "Chat ended." << std::endl;
    return 0;
}

int runChatModelEvaluationExample() {
    const AppPaths paths;

    std::cout << "\n========================================" << std::endl;
    std::cout << "        Chat Model Evaluation" << std::endl;
    std::cout << "========================================\n" << std::endl;

    NKS_Tokenizer tokenizer = createBpeTokenizer();
    if (!loadOrTrainBpeModelOrReport(
            tokenizer,
            paths.bpeTrainingPath,
            paths.bpeModelPath,
            paths.mergedTxtCorpusPath)) {
        return 1;
    }

    ChatNgramModel chatModel;
    if (!chatModel.load(paths.chatModelPath) || chatModel.empty()) {
        std::cerr << "No real corpus chat model found at " << paths.chatModelPath
                  << ". Run option 4 first." << std::endl;
        return 1;
    }

    std::cout << "Loaded: " << paths.chatModelPath << std::endl;
    std::cout << "Observed token pairs: " << chatModel.observedPairs << std::endl;
    std::cout << "Observed token triples: " << chatModel.observedTriples << std::endl;
    std::cout << "Transition rows: " << chatModel.transitions.size() << std::endl;
    std::cout << "Pair transition rows: " << chatModel.pairTransitions.size() << std::endl;

    printChatModelSample(chatModel, tokenizer, "how are you");
    printChatModelSample(chatModel, tokenizer, "what is your name");
    printChatModelSample(chatModel, tokenizer, "tell me about language models");
    printChatModelSample(chatModel, tokenizer, "write a short answer");

    std::cout << "\nEnter custom prompts. Press Ctrl+C or EOF to stop.\n" << std::endl;
    std::string prompt;
    while (true) {
        std::cout << "Prompt> ";
        if (!std::getline(std::cin, prompt)) {
            std::cout << std::endl;
            break;
        }
        if (prompt.empty()) {
            continue;
        }
        printChatModelSample(chatModel, tokenizer, prompt);
    }

    return 0;
}
