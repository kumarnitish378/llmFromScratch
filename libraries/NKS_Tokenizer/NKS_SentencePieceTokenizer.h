#ifndef NKS_SENTENCEPIECE_TOKENIZER_H
#define NKS_SENTENCEPIECE_TOKENIZER_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class NKS_SentencePieceTokenizer {
public:
    struct TrainingConfig {
        std::size_t targetVocabSize = 8000;
        std::size_t maxPieceChars = 8;
        std::size_t trainingLineLimit = 50000;
        bool lowercase = true;
        bool splitCamelCase = true;
    };

    enum class EncodeFallback {
        kUnknownPiece,
        kSingleChar
    };

    NKS_SentencePieceTokenizer();

    NKS_SentencePieceTokenizer& setTargetVocabSize(std::size_t size);
    NKS_SentencePieceTokenizer& setMaxPieceChars(std::size_t size);
    NKS_SentencePieceTokenizer& setTrainingLineLimit(std::size_t limit);
    NKS_SentencePieceTokenizer& setLowercase(bool enabled);
    NKS_SentencePieceTokenizer& setSplitCamelCase(bool enabled);
    NKS_SentencePieceTokenizer& setTrainingConfig(const TrainingConfig& config);

    bool trainFromFile(const std::string& corpusPath);

    std::vector<std::string> encode(const std::string& text) const;
    std::vector<int> encodeToIds(const std::string& text) const;
    std::string decode(const std::vector<std::string>& pieces) const;
    std::string decodeFromIds(const std::vector<int>& ids) const;

    std::size_t vocabularySize() const;

private:
    struct Utf8Char {
        uint32_t codepoint;
        std::string bytes;
    };

    static constexpr const char* kSpaceMarker = "\xE2\x96\x81"; // ▁

    static std::vector<Utf8Char> splitUtf8Chars(const std::string& text);
    static char toLowerAscii(char c);
    static bool isAsciiLower(uint32_t cp);
    static bool isAsciiUpper(uint32_t cp);
    static bool isUnicodeWhitespace(uint32_t cp);
    static std::string trimAsciiSpaces(const std::string& s);

    std::string normalizeInputForTraining(const std::string& text) const;
    std::string normalizeInputForEncoding(const std::string& text) const;
    void resetVocabulary();
    void buildSeedCharacters(const std::vector<std::string>& corpusLines);
    void addFrequentSubstringCandidates(const std::vector<std::string>& corpusLines);

    std::unordered_map<std::string, int> pieceToId_;
    std::vector<std::string> idToPiece_;
    std::unordered_map<std::string, double> pieceLogProb_;

    TrainingConfig config_;
    int unknownId_ = 0;

    static constexpr std::size_t kMinPieceChars = 2;
    static constexpr std::size_t kMinTargetVocab = 64;
    static constexpr std::size_t kMinLineLimit = 1;
    static constexpr std::size_t kUtf8ByteReserveFactor = 4;
    static constexpr std::size_t kEncodingReservePadding = 8;
    static constexpr double kFallbackCost = 20.0;
    static constexpr int kMinNgramFrequency = 2;
    static constexpr double kUnknownLogProb = -20.0;
};

#endif
