#ifndef NKS_TOKENIZER_H
#define NKS_TOKENIZER_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class NKS_Tokenizer {
public:
    NKS_Tokenizer();
    explicit NKS_Tokenizer(const std::string& vocabularyPath);

    NKS_Tokenizer& setLowercase(bool enabled);
    NKS_Tokenizer& setSplitOnPunctuation(bool enabled);
    NKS_Tokenizer& setKeepPunctuation(bool enabled);
    NKS_Tokenizer& setUnknownToken(const std::string& token);
    NKS_Tokenizer& setPreserveUnknownTokens(bool enabled);

    bool loadVocabulary(const std::string& vocabularyPath);
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokenIds) const;

    std::size_t vocabularySize() const;
    std::size_t estimateModelTokensApprox(const std::string& text) const;

private:
    struct Utf8Char {
        uint32_t codepoint;
        std::string bytes;
    };

    static std::vector<Utf8Char> splitUtf8Chars(const std::string& text);
    static std::string normalizeToken(const std::string& token, bool lowercase);
    static bool isUnicodeWhitespace(uint32_t cp);
    static bool isUnicodePunctuation(uint32_t cp);
    static bool isAsciiAlphaNum(uint32_t cp);
    static char toLowerAscii(char c);
    bool isSpecialToken(const std::string& token) const;

    std::unordered_map<std::string, int> tokenToId_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> dynamicTokenToId_;
    std::unordered_map<int, std::string> dynamicIdToToken_;
    int nextDynamicId_ = 1;
    int unknownTokenId_ = 0;
    bool lowercase_ = true;
    bool splitOnPunctuation_ = true;
    bool keepPunctuation_ = true;
    bool preserveUnknownTokens_ = true;
    std::string unknownToken_ = "<unk>";
};

#endif
