#include <iostream>
#include <string>
#include <vector>

#include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"

namespace {
struct TokenizationResult {
    std::vector<std::string> pieces;
    std::vector<int> tokenIds;
    std::string decodedText;
    std::size_t approxModelTokenCount = 0;
};

NKS_Tokenizer createTokenizer() {
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

bool loadVocabularyOrReport(NKS_Tokenizer& tokenizer, const std::string& vocabularyPath) {
    if (tokenizer.loadVocabulary(vocabularyPath)) {
        return true;
    }

    std::cerr << "Failed to load vocabulary file: " << vocabularyPath << std::endl;
    return false;
}

TokenizationResult runTokenizationPipeline(NKS_Tokenizer& tokenizer, const std::string& text) {
    TokenizationResult result;
    result.pieces = tokenizer.tokenize(text);
    result.tokenIds = tokenizer.encode(text);
    result.decodedText = tokenizer.decode(result.tokenIds);
    result.approxModelTokenCount = tokenizer.estimateModelTokensApprox(text);
    return result;
}

void printPieces(const std::vector<std::string>& pieces) {
    std::cout << "Tokenizer pieces: ";
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        std::cout << "[" << pieces[i] << "]";
        if (i + 1 < pieces.size()) {
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

void printSummary(const NKS_Tokenizer& tokenizer, const std::string& inputText, const TokenizationResult& result) {
    std::cout << "Vocabulary size: " << tokenizer.vocabularySize() << std::endl;
    std::cout << "Input text: " << inputText << std::endl;
    std::cout << "Approx model token count: " << result.approxModelTokenCount << std::endl;
    printPieces(result.pieces);
    printTokenIds(result.tokenIds);
    std::cout << "Decoded text: " << result.decodedText << std::endl;
}

std::string readInputTextFromTerminal() {
    std::cout << "Enter text to tokenize: ";
    std::string inputText;
    std::getline(std::cin, inputText);
    return inputText;
}
} // namespace

int main() {
    const std::string vocabularyPath = "Data/words.txt";

    NKS_Tokenizer tokenizer = createTokenizer();
    if (!loadVocabularyOrReport(tokenizer, vocabularyPath)) {
        return 1;
    }

    const std::string inputText = readInputTextFromTerminal();
    if (inputText.empty()) {
        std::cerr << "No input provided." << std::endl;
        return 1;
    }

    const TokenizationResult result = runTokenizationPipeline(tokenizer, inputText);
    printSummary(tokenizer, inputText, result);
    return 0;
}
