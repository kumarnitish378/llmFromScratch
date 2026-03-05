#include "NKS_Tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

NKS_Tokenizer::NKS_Tokenizer() {
    idToToken_.push_back(unknownToken_);
    tokenToId_[unknownToken_] = 0;
    unknownTokenId_ = 0;
    nextDynamicId_ = 1;
}

NKS_Tokenizer::NKS_Tokenizer(const std::string& vocabularyPath) : NKS_Tokenizer() {
    loadVocabulary(vocabularyPath);
}

NKS_Tokenizer& NKS_Tokenizer::setLowercase(bool enabled) {
    lowercase_ = enabled;
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setSplitOnPunctuation(bool enabled) {
    splitOnPunctuation_ = enabled;
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setKeepPunctuation(bool enabled) {
    keepPunctuation_ = enabled;
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setUnknownToken(const std::string& token) {
    const std::string candidate = normalizeToken(token, lowercase_);
    if (!candidate.empty()) {
        unknownToken_ = candidate;
    }
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setPreserveUnknownTokens(bool enabled) {
    preserveUnknownTokens_ = enabled;
    return *this;
}

bool NKS_Tokenizer::loadVocabulary(const std::string& vocabularyPath) {
    std::ifstream inFile(vocabularyPath);
    if (!inFile.is_open()) {
        return false;
    }

    tokenToId_.clear();
    idToToken_.clear();
    dynamicTokenToId_.clear();
    dynamicIdToToken_.clear();

    idToToken_.push_back(unknownToken_);
    tokenToId_[unknownToken_] = 0;
    unknownTokenId_ = 0;

    std::string line;
    while (std::getline(inFile, line)) {
        const std::string token = normalizeToken(line, lowercase_);
        if (token.empty()) {
            continue;
        }

        if (tokenToId_.find(token) != tokenToId_.end()) {
            continue;
        }

        const int tokenId = static_cast<int>(idToToken_.size());
        idToToken_.push_back(token);
        tokenToId_[token] = tokenId;
    }

    nextDynamicId_ = static_cast<int>(idToToken_.size());
    return true;
}

std::vector<std::string> NKS_Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    const std::vector<Utf8Char> chars = splitUtf8Chars(text);
    std::string currentToken;

    auto pushCurrentToken = [&]() {
        if (currentToken.empty()) {
            return;
        }
        const std::string normalized = normalizeToken(currentToken, lowercase_);
        if (!normalized.empty()) {
            tokens.push_back(normalized);
        }
        currentToken.clear();
    };

    for (const Utf8Char& ch : chars) {
        if (isUnicodeWhitespace(ch.codepoint)) {
            pushCurrentToken();
            continue;
        }

        if (splitOnPunctuation_ && isUnicodePunctuation(ch.codepoint)) {
            pushCurrentToken();
            if (keepPunctuation_) {
                const std::string punctuationToken = normalizeToken(ch.bytes, lowercase_);
                if (!punctuationToken.empty()) {
                    tokens.push_back(punctuationToken);
                }
            }
            continue;
        }

        if (lowercase_ && ch.codepoint <= 127) {
            currentToken.push_back(toLowerAscii(ch.bytes[0]));
        } else {
            currentToken.append(ch.bytes);
        }
    }

    pushCurrentToken();

    return tokens;
}

std::vector<int> NKS_Tokenizer::encode(const std::string& text) {
    std::vector<int> tokenIds;
    const std::vector<std::string> tokens = tokenize(text);
    tokenIds.reserve(tokens.size());

    for (const std::string& token : tokens) {
        const auto it = tokenToId_.find(token);
        if (it != tokenToId_.end()) {
            tokenIds.push_back(it->second);
            continue;
        }

        if (!preserveUnknownTokens_) {
            tokenIds.push_back(unknownTokenId_);
            continue;
        }

        const auto dynamicIt = dynamicTokenToId_.find(token);
        if (dynamicIt != dynamicTokenToId_.end()) {
            tokenIds.push_back(dynamicIt->second);
            continue;
        }

        const int newDynamicId = nextDynamicId_++;
        dynamicTokenToId_[token] = newDynamicId;
        dynamicIdToToken_[newDynamicId] = token;
        tokenIds.push_back(newDynamicId);
    }

    return tokenIds;
}

std::string NKS_Tokenizer::decode(const std::vector<int>& tokenIds) const {
    std::ostringstream oss;
    bool prevIsConnectorPunctuation = false;

    for (std::size_t i = 0; i < tokenIds.size(); ++i) {
        const int id = tokenIds[i];
        std::string token = idToToken_[unknownTokenId_];
        if (id >= 0 && static_cast<std::size_t>(id) < idToToken_.size()) {
            token = idToToken_[id];
        } else {
            const auto dynamicIt = dynamicIdToToken_.find(id);
            if (dynamicIt != dynamicIdToToken_.end()) {
                token = dynamicIt->second;
            }
        }

        bool isSinglePunctuation = false;
        bool isConnectorPunctuation = false;
        if (!token.empty()) {
            const std::vector<Utf8Char> chars = splitUtf8Chars(token);
            if (chars.size() == 1 && isUnicodePunctuation(chars[0].codepoint)) {
                isSinglePunctuation = true;
                const std::string& punct = chars[0].bytes;
                isConnectorPunctuation = (punct == "-" || punct == "'" || punct == "/" || punct == "_");
            }
        }

        if (i > 0 && !isSinglePunctuation && !prevIsConnectorPunctuation) {
            oss << ' ';
        }
        oss << token;
        prevIsConnectorPunctuation = isConnectorPunctuation;
    }

    return oss.str();
}

std::size_t NKS_Tokenizer::vocabularySize() const {
    return idToToken_.size();
}

std::size_t NKS_Tokenizer::estimateModelTokensApprox(const std::string& text) const {
    // Practical estimate from OpenAI guidance: ~1 token ~= 4 English chars.
    return static_cast<std::size_t>(std::ceil(static_cast<double>(text.size()) / 4.0));
}

std::vector<NKS_Tokenizer::Utf8Char> NKS_Tokenizer::splitUtf8Chars(const std::string& text) {
    std::vector<Utf8Char> out;
    out.reserve(text.size());

    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        std::size_t width = 1;
        uint32_t cp = lead;

        if ((lead & 0xE0u) == 0xC0u && i + 1 < text.size()) {
            width = 2;
            cp = ((lead & 0x1Fu) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3Fu);
        } else if ((lead & 0xF0u) == 0xE0u && i + 2 < text.size()) {
            width = 3;
            cp = ((lead & 0x0Fu) << 12) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6) |
                 (static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
        } else if ((lead & 0xF8u) == 0xF0u && i + 3 < text.size()) {
            width = 4;
            cp = ((lead & 0x07u) << 18) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3Fu) << 6) |
                 (static_cast<unsigned char>(text[i + 3]) & 0x3Fu);
        }

        out.push_back({cp, text.substr(i, width)});
        i += width;
    }

    return out;
}

std::string NKS_Tokenizer::normalizeToken(const std::string& token, bool lowercase) {
    std::string normalized;
    normalized.reserve(token.size());

    for (unsigned char ch : token) {
        if (ch == '\r' || ch == '\n' || ch == '\t' || ch == '\0') {
            continue;
        }

        if (lowercase && ch <= 127) {
            normalized.push_back(toLowerAscii(static_cast<char>(ch)));
        } else {
            normalized.push_back(static_cast<char>(ch));
        }
    }

    const auto first = normalized.find_first_not_of(' ');
    if (first == std::string::npos) {
        return "";
    }

    const auto last = normalized.find_last_not_of(' ');
    return normalized.substr(first, last - first + 1);
}

bool NKS_Tokenizer::isUnicodeWhitespace(uint32_t cp) {
    if (cp <= 127) {
        return std::isspace(static_cast<unsigned char>(cp)) != 0;
    }
    return cp == 0x00A0u || cp == 0x1680u || cp == 0x180Eu || cp == 0x2007u || cp == 0x202Fu || cp == 0x3000u;
}

bool NKS_Tokenizer::isUnicodePunctuation(uint32_t cp) {
    if (cp <= 127) {
        return std::ispunct(static_cast<unsigned char>(cp)) != 0;
    }
    if ((cp >= 0x2000u && cp <= 0x206Fu) || (cp >= 0x3000u && cp <= 0x303Fu) || (cp >= 0xFF01u && cp <= 0xFF65u)) {
        return true;
    }
    return false;
}

bool NKS_Tokenizer::isAsciiAlphaNum(uint32_t cp) {
    return cp <= 127 && (std::isalnum(static_cast<unsigned char>(cp)) != 0);
}

char NKS_Tokenizer::toLowerAscii(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool NKS_Tokenizer::isSpecialToken(const std::string& token) const {
    return !token.empty() && tokenToId_.find(token) != tokenToId_.end();
}
