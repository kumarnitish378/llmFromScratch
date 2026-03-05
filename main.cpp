#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "libraries/NKS_Tokenizer/NKS_SentencePieceTokenizer.h"
#include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"

namespace {
struct AppPaths {
    std::string vocabularyPath = "Data/words.txt";
    std::string bpeModelPath = "Metadata/bpe_model.bin";
};

struct BpeRuntimeConfig {
    std::size_t mergeOps = 600;
    std::size_t trainingWordLimit = 25000;
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

std::string formatPieceForTerminal(const std::string& piece) {
    const std::string marker = "\xE2\x96\x81"; // ▁
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
    const std::string marker = "\xE2\x96\x81"; // ▁
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

struct TokenizationResult {
    std::vector<std::string> pieces;
    std::vector<int> tokenIds;
    std::string decodedText;
    std::size_t approxModelTokenCount = 0;
    std::size_t vocabularySize = 0;
};

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

    tokenizer
        .setTrainingConfig(trainingCfg);
    return tokenizer;
}

bool loadOrTrainBpeModelOrReport(
    NKS_Tokenizer& tokenizer,
    const std::string& vocabularyPath,
    const std::string& modelPath) {
    if (tokenizer.loadModel(modelPath)) {
        std::cout << "Loaded BPE model from metadata: " << modelPath << std::endl;
        return true;
    }

    std::cout << "Metadata model not found/invalid. Training BPE model..." << std::endl;
    if (!tokenizer.loadVocabulary(vocabularyPath)) {
        std::cerr << "Failed to train BPE model from vocabulary file: " << vocabularyPath << std::endl;
        return false;
    }

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
    result.approxModelTokenCount = static_cast<std::size_t>(std::ceil(static_cast<double>(text.size()) / kApproxCharsPerToken));
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

    if (mode == "sp") {
        return TokenizerMode::kSentencePiece;
    }
    if (mode == "sentencepiece") {
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

int main() {
    const AppPaths paths;

    const TokenizerMode mode = parseTokenizerMode();
    const std::string inputText = readInputTextFromTerminal();
    if (inputText.empty()) {
        std::cerr << "No input provided." << std::endl;
        return 1;
    }

    if (mode == TokenizerMode::kBpe) {
        NKS_Tokenizer tokenizer = createBpeTokenizer();
        if (!loadOrTrainBpeModelOrReport(tokenizer, paths.vocabularyPath, paths.bpeModelPath)) {
            return 1;
        }

        const TokenizationResult result = runBpePipeline(tokenizer, inputText);
        printSummary("bpe", inputText, result);
        return 0;
    }

    if (mode == TokenizerMode::kSentencePiece) {
        NKS_SentencePieceTokenizer tokenizer = createSentencePieceTokenizer();
        if (!trainSentencePieceOrReport(tokenizer, paths.vocabularyPath)) {
            return 1;
        }

        const TokenizationResult result = runSentencePiecePipeline(tokenizer, inputText);
        printSummary("sentencepiece", inputText, result);
        return 0;
    }

    std::cerr << "Unsupported mode. Use 'bpe' or 'sentencepiece'." << std::endl;
    return 1;
}
