#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "libraries/NKS_Tokenizer/NKS_SentencePieceTokenizer.h"
#include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"

namespace {
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
    NKS_Tokenizer tokenizer;
    tokenizer
        .setLowercase(true)
        .setSplitOnPunctuation(true)
        .setKeepPunctuation(true)
        .setSplitCamelCase(true)
        .setBpeMergeOps(600)
        .setTrainingWordLimit(25000)
        .setPreserveUnknownTokens(true);
    return tokenizer;
}

NKS_SentencePieceTokenizer createSentencePieceTokenizer() {
    NKS_SentencePieceTokenizer tokenizer;
    tokenizer
        .setLowercase(true)
        .setSplitCamelCase(true)
        .setTargetVocabSize(2000)
        .setMaxPieceChars(8)
        .setTrainingLineLimit(25000);
    return tokenizer;
}

bool loadBpeVocabularyOrReport(NKS_Tokenizer& tokenizer, const std::string& vocabularyPath) {
    if (tokenizer.loadVocabulary(vocabularyPath)) {
        return true;
    }
    std::cerr << "Failed to load BPE vocabulary file: " << vocabularyPath << std::endl;
    return false;
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
    result.approxModelTokenCount = static_cast<std::size_t>(std::ceil(static_cast<double>(text.size()) / 4.0));
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

std::string readTokenizerMode() {
    std::cout << "Choose tokenizer mode [bpe/sentencepiece] (default=bpe): ";
    std::string mode;
    std::getline(std::cin, mode);
    if (mode.empty()) {
        return "bpe";
    }

    for (char& c : mode) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (mode == "sp") {
        return "sentencepiece";
    }
    return mode;
}

std::string readInputTextFromTerminal() {
    std::cout << "Enter text to tokenize: ";
    std::string inputText;
    std::getline(std::cin, inputText);
    return inputText;
}
} // namespace

int main() {
    const std::string dataPath = "Data/words.txt";

    const std::string mode = readTokenizerMode();
    const std::string inputText = readInputTextFromTerminal();
    if (inputText.empty()) {
        std::cerr << "No input provided." << std::endl;
        return 1;
    }

    if (mode == "bpe") {
        NKS_Tokenizer tokenizer = createBpeTokenizer();
        if (!loadBpeVocabularyOrReport(tokenizer, dataPath)) {
            return 1;
        }

        const TokenizationResult result = runBpePipeline(tokenizer, inputText);
        printSummary("bpe", inputText, result);
        return 0;
    }

    if (mode == "sentencepiece") {
        NKS_SentencePieceTokenizer tokenizer = createSentencePieceTokenizer();
        if (!trainSentencePieceOrReport(tokenizer, dataPath)) {
            return 1;
        }

        const TokenizationResult result = runSentencePiecePipeline(tokenizer, inputText);
        printSummary("sentencepiece", inputText, result);
        return 0;
    }

    std::cerr << "Unsupported mode: " << mode << ". Use 'bpe' or 'sentencepiece'." << std::endl;
    return 1;
}
