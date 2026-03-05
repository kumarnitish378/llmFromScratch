#include "NKS_Tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace {
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        const std::size_t h1 = std::hash<std::string>{}(p.first);
        const std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct WordSymbols {
    std::vector<std::string> symbols;
    int freq;
};

bool endsWith(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}
} // namespace

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

NKS_Tokenizer& NKS_Tokenizer::setSplitCamelCase(bool enabled) {
    splitCamelCase_ = enabled;
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

NKS_Tokenizer& NKS_Tokenizer::setBpeMergeOps(std::size_t mergeOps) {
    if (mergeOps >= kMinValidCount) {
        trainingConfig_.mergeOps = mergeOps;
    }
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setTrainingWordLimit(std::size_t maxWords) {
    if (maxWords >= kMinValidCount) {
        trainingConfig_.trainingWordLimit = maxWords;
    }
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setShowTrainingProgress(bool enabled) {
    trainingConfig_.showProgress = enabled;
    return *this;
}

NKS_Tokenizer& NKS_Tokenizer::setTrainingConfig(const BpeTrainingConfig& config) {
    trainingConfig_ = config;
    if (trainingConfig_.mergeOps < kMinValidCount) {
        trainingConfig_.mergeOps = kMinValidCount;
    }
    if (trainingConfig_.trainingWordLimit < kMinValidCount) {
        trainingConfig_.trainingWordLimit = kMinValidCount;
    }
    return *this;
}

bool NKS_Tokenizer::loadVocabulary(const std::string& vocabularyPath) {
    std::ifstream inFile(vocabularyPath);
    if (!inFile.is_open()) {
        return false;
    }

    std::vector<std::string> words;
    words.reserve(trainingConfig_.trainingWordLimit);

    std::string line;
    while (std::getline(inFile, line)) {
        const std::string token = normalizeToken(line, lowercase_);
        if (token.empty()) {
            continue;
        }

        words.push_back(token);
        if (words.size() >= trainingConfig_.trainingWordLimit) {
            break;
        }

        if (trainingConfig_.showProgress && (words.size() % kProgressWordInterval == 0)) {
            std::cout << "[BPE] Loaded " << words.size() << " training words..." << std::endl;
        }
    }

    if (trainingConfig_.showProgress) {
        std::cout << "[BPE] Starting training with " << words.size() << " words and " << trainingConfig_.mergeOps
                  << " merge steps max..." << std::endl;
    }
    trainBpe(words);
    rebuildIdMapsFromSubwords();

    dynamicTokenToId_.clear();
    dynamicIdToToken_.clear();
    nextDynamicId_ = static_cast<int>(idToToken_.size());
    return true;
}

bool NKS_Tokenizer::saveModel(const std::string& modelPath) const {
    std::ofstream out(modelPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    std::vector<std::string> sortedSubwords(subwordVocab_.begin(), subwordVocab_.end());
    std::sort(sortedSubwords.begin(), sortedSubwords.end());

    const char magic[8] = {'N', 'K', 'S', '_', 'B', 'P', 'E', '1'};
    out.write(magic, sizeof(magic));

    uint32_t unknownLen = static_cast<uint32_t>(unknownToken_.size());
    uint32_t maxSub = static_cast<uint32_t>(maxSubwordChars_);
    uint32_t tokenCount = static_cast<uint32_t>(sortedSubwords.size());

    out.write(reinterpret_cast<const char*>(&unknownLen), sizeof(unknownLen));
    out.write(unknownToken_.data(), unknownToken_.size());
    out.write(reinterpret_cast<const char*>(&maxSub), sizeof(maxSub));
    out.write(reinterpret_cast<const char*>(&tokenCount), sizeof(tokenCount));

    for (const std::string& token : sortedSubwords) {
        uint32_t len = static_cast<uint32_t>(token.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(token.data(), token.size());
    }

    return out.good();
}

bool NKS_Tokenizer::loadModel(const std::string& modelPath) {
    std::ifstream in(modelPath, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'N', 'K', 'S', '_', 'B', 'P', 'E', '1'};
    if (!in.good() || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        return false;
    }

    uint32_t unknownLen = 0;
    in.read(reinterpret_cast<char*>(&unknownLen), sizeof(unknownLen));
    if (!in.good()) {
        return false;
    }
    std::string loadedUnknownToken(unknownLen, '\0');
    if (unknownLen > 0) {
        in.read(&loadedUnknownToken[0], unknownLen);
    }
    if (!in.good()) {
        return false;
    }

    uint32_t maxSubwordRaw = 0;
    in.read(reinterpret_cast<char*>(&maxSubwordRaw), sizeof(maxSubwordRaw));
    if (!in.good()) {
        return false;
    }

    uint32_t tokenCountRaw = 0;
    in.read(reinterpret_cast<char*>(&tokenCountRaw), sizeof(tokenCountRaw));
    if (!in.good()) {
        return false;
    }

    const std::size_t loadedMaxSubwordChars = static_cast<std::size_t>(maxSubwordRaw);
    const std::size_t tokenCount = static_cast<std::size_t>(tokenCountRaw);

    std::unordered_set<std::string> loadedSubwords;
    loadedSubwords.reserve(tokenCount);

    for (std::size_t i = 0; i < tokenCount; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!in.good()) {
            return false;
        }

        std::string token(len, '\0');
        if (len > 0) {
            in.read(&token[0], len);
        }
        if (!in.good()) {
            return false;
        }
        if (!token.empty()) {
            loadedSubwords.insert(token);
        }
    }

    unknownToken_ = loadedUnknownToken.empty() ? kUnknownTokenLiteral : loadedUnknownToken;
    maxSubwordChars_ = std::max<std::size_t>(1, loadedMaxSubwordChars);
    subwordVocab_ = std::move(loadedSubwords);

    rebuildIdMapsFromSubwords();

    dynamicTokenToId_.clear();
    dynamicIdToToken_.clear();
    nextDynamicId_ = static_cast<int>(idToToken_.size());
    return true;
}

std::vector<std::string> NKS_Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> output;
    const std::vector<PreToken> parts = preTokenize(text);

    for (const PreToken& part : parts) {
        if (part.text.empty()) {
            continue;
        }

        if (part.isPunctuation) {
            output.push_back(normalizeToken(part.text, lowercase_));
            continue;
        }

        std::vector<std::string> pieces = subwordTokenizeWord(part.text);
        output.insert(output.end(), pieces.begin(), pieces.end());
    }

    return output;
}

std::vector<int> NKS_Tokenizer::encode(const std::string& text) {
    const std::vector<std::string> pieces = tokenize(text);
    std::vector<int> tokenIds;
    tokenIds.reserve(pieces.size());

    for (const std::string& piece : pieces) {
        const auto it = tokenToId_.find(piece);
        if (it != tokenToId_.end()) {
            tokenIds.push_back(it->second);
            continue;
        }

        if (!preserveUnknownTokens_) {
            tokenIds.push_back(unknownTokenId_);
            continue;
        }

        const auto dyn = dynamicTokenToId_.find(piece);
        if (dyn != dynamicTokenToId_.end()) {
            tokenIds.push_back(dyn->second);
            continue;
        }

        const int newId = nextDynamicId_++;
        dynamicTokenToId_[piece] = newId;
        dynamicIdToToken_[newId] = piece;
        tokenIds.push_back(newId);
    }

    return tokenIds;
}

std::string NKS_Tokenizer::decode(const std::vector<int>& tokenIds) const {
    std::ostringstream oss;
    bool isStart = true;

    for (int id : tokenIds) {
        std::string token = unknownToken_;

        if (id >= 0 && static_cast<std::size_t>(id) < idToToken_.size()) {
            token = idToToken_[id];
        } else {
            const auto dyn = dynamicIdToToken_.find(id);
            if (dyn != dynamicIdToToken_.end()) {
                token = dyn->second;
            }
        }

        const bool isContinuation = token.rfind("##", 0) == 0;
        const std::string clean = isContinuation ? token.substr(2) : token;
        const bool isPunctuation = splitUtf8Chars(clean).size() == 1 && isUnicodePunctuation(splitUtf8Chars(clean)[0].codepoint);

        if (isStart) {
            oss << clean;
            isStart = false;
            continue;
        }

        if (isContinuation || isPunctuation || isConnectorPunctuation(clean)) {
            oss << clean;
        } else {
            oss << ' ' << clean;
        }
    }

    return oss.str();
}

std::size_t NKS_Tokenizer::vocabularySize() const {
    return idToToken_.size();
}

std::size_t NKS_Tokenizer::estimateModelTokensApprox(const std::string& text) const {
    return static_cast<std::size_t>(std::ceil(static_cast<double>(text.size()) / kApproxCharsPerToken));
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

bool NKS_Tokenizer::isAsciiLower(uint32_t cp) {
    return cp >= 'a' && cp <= 'z';
}

bool NKS_Tokenizer::isAsciiUpper(uint32_t cp) {
    return cp >= 'A' && cp <= 'Z';
}

char NKS_Tokenizer::toLowerAscii(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

std::size_t NKS_Tokenizer::utf8CharCount(const std::string& text) {
    return splitUtf8Chars(text).size();
}

std::vector<NKS_Tokenizer::PreToken> NKS_Tokenizer::preTokenize(const std::string& text) const {
    std::vector<PreToken> out;
    std::vector<Utf8Char> chars = splitUtf8Chars(text);

    std::string current;
    uint32_t prevCp = 0;
    bool hasPrev = false;

    auto flushWord = [&]() {
        if (current.empty()) {
            return;
        }
        out.push_back({current, false});
        current.clear();
        hasPrev = false;
    };

    for (const Utf8Char& ch : chars) {
        if (isUnicodeWhitespace(ch.codepoint)) {
            flushWord();
            continue;
        }

        if (splitOnPunctuation_ && isUnicodePunctuation(ch.codepoint)) {
            flushWord();
            if (keepPunctuation_) {
                out.push_back({ch.bytes, true});
            }
            continue;
        }

        if (splitCamelCase_ && hasPrev && isAsciiLower(prevCp) && isAsciiUpper(ch.codepoint)) {
            flushWord();
        }

        current.append(ch.bytes);
        prevCp = ch.codepoint;
        hasPrev = true;
    }

    flushWord();
    return out;
}

std::vector<std::string> NKS_Tokenizer::subwordTokenizeWord(const std::string& word) const {
    std::vector<std::string> pieces;
    const std::string normalized = normalizeToken(word, lowercase_);
    if (normalized.empty()) {
        return pieces;
    }

    const std::vector<Utf8Char> chars = splitUtf8Chars(normalized);
    if (chars.empty()) {
        return pieces;
    }

    std::size_t index = 0;
    bool isFirst = true;

    while (index < chars.size()) {
        std::string matched;
        std::size_t matchedLen = 0;

        const std::size_t maxLen = std::min(maxSubwordChars_, chars.size() - index);
        for (std::size_t len = maxLen; len >= 1; --len) {
            std::string candidate;
            for (std::size_t k = 0; k < len; ++k) {
                candidate.append(chars[index + k].bytes);
            }

            if (subwordVocab_.find(candidate) != subwordVocab_.end()) {
                matched = candidate;
                matchedLen = len;
                break;
            }

            if (len == 1) {
                break;
            }
        }

        if (matched.empty()) {
            matched = chars[index].bytes;
            matchedLen = 1;
        }

        if (!isFirst) {
            matched = "##" + matched;
        }

        pieces.push_back(matched);
        index += matchedLen;
        isFirst = false;
    }

    return pieces;
}

void NKS_Tokenizer::trainBpe(const std::vector<std::string>& words) {
    std::unordered_map<std::string, int> wordFreq;
    for (const std::string& w : words) {
        ++wordFreq[w];
    }

    std::vector<WordSymbols> corpus;
    corpus.reserve(wordFreq.size());

    subwordVocab_.clear();
    maxSubwordChars_ = 1;

    for (const auto& it : wordFreq) {
        const std::vector<Utf8Char> chars = splitUtf8Chars(it.first);
        if (chars.empty()) {
            continue;
        }

        WordSymbols ws;
        ws.freq = it.second;
        ws.symbols.reserve(chars.size() + 1);
        for (const Utf8Char& ch : chars) {
            ws.symbols.push_back(ch.bytes);
            subwordVocab_.insert(ch.bytes);
        }
        ws.symbols.push_back(kWordBoundaryMarker);
        corpus.push_back(std::move(ws));
    }

    for (std::size_t step = 0; step < trainingConfig_.mergeOps; ++step) {
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pairCounts;

        for (const WordSymbols& ws : corpus) {
            if (ws.symbols.size() < 2) {
                continue;
            }
            for (std::size_t i = 0; i + 1 < ws.symbols.size(); ++i) {
                ++pairCounts[{ws.symbols[i], ws.symbols[i + 1]}];
            }
        }

        int bestCount = 0;
        std::pair<std::string, std::string> bestPair;
        bool found = false;

        for (const auto& kv : pairCounts) {
            if (kv.second > bestCount) {
                bestCount = kv.second;
                bestPair = kv.first;
                found = true;
            }
        }

        if (!found || bestCount < kMinFrequentPairCount) {
            if (trainingConfig_.showProgress) {
                std::cout << "[BPE] Early stop at step " << step << " (no frequent pairs)." << std::endl;
            }
            break;
        }

        const std::string merged = bestPair.first + bestPair.second;

        for (WordSymbols& ws : corpus) {
            if (ws.symbols.size() < 2) {
                continue;
            }

            std::vector<std::string> mergedSymbols;
            mergedSymbols.reserve(ws.symbols.size());

            std::size_t i = 0;
            while (i < ws.symbols.size()) {
                if (i + 1 < ws.symbols.size() && ws.symbols[i] == bestPair.first && ws.symbols[i + 1] == bestPair.second) {
                    mergedSymbols.push_back(merged);
                    i += 2;
                } else {
                    mergedSymbols.push_back(ws.symbols[i]);
                    ++i;
                }
            }

            ws.symbols = std::move(mergedSymbols);
        }

        if (trainingConfig_.showProgress &&
            ((step + 1) % kProgressMergeInterval == 0 || step + 1 == trainingConfig_.mergeOps)) {
            const double pct = (100.0 * static_cast<double>(step + 1)) / static_cast<double>(trainingConfig_.mergeOps);
            std::cout << "[BPE] Merge progress: " << (step + 1) << "/" << trainingConfig_.mergeOps
                      << " (" << static_cast<int>(pct) << "%)" << std::endl;
        }
    }

    for (const WordSymbols& ws : corpus) {
        for (const std::string& sym : ws.symbols) {
            if (sym == kWordBoundaryMarker) {
                continue;
            }

            std::string normalizedSymbol = sym;
            if (endsWith(normalizedSymbol, kWordBoundaryMarker)) {
                normalizedSymbol = normalizedSymbol.substr(
                    0, normalizedSymbol.size() - std::char_traits<char>::length(kWordBoundaryMarker));
            }

            if (normalizedSymbol.empty()) {
                continue;
            }

            subwordVocab_.insert(normalizedSymbol);
            maxSubwordChars_ = std::max(maxSubwordChars_, utf8CharCount(normalizedSymbol));
        }
    }
}

void NKS_Tokenizer::rebuildIdMapsFromSubwords() {
    tokenToId_.clear();
    idToToken_.clear();

    idToToken_.push_back(unknownToken_);
    tokenToId_[unknownToken_] = 0;
    unknownTokenId_ = 0;

    std::vector<std::string> sorted(subwordVocab_.begin(), subwordVocab_.end());
    std::sort(sorted.begin(), sorted.end(), [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) {
            return a.size() > b.size();
        }
        return a < b;
    });

    for (const std::string& s : sorted) {
        if (tokenToId_.find(s) == tokenToId_.end()) {
            const int id = static_cast<int>(idToToken_.size());
            idToToken_.push_back(s);
            tokenToId_[s] = id;
        }

        const std::string continuation = std::string(kContinuationPrefix) + s;
        if (tokenToId_.find(continuation) == tokenToId_.end()) {
            const int id = static_cast<int>(idToToken_.size());
            idToToken_.push_back(continuation);
            tokenToId_[continuation] = id;
        }
    }

    const std::string punct = ".,!?:;()[]{}<>-_/+'\"";
    for (char c : punct) {
        std::string p(1, c);
        if (tokenToId_.find(p) == tokenToId_.end()) {
            const int id = static_cast<int>(idToToken_.size());
            idToToken_.push_back(p);
            tokenToId_[p] = id;
        }
    }
}

bool NKS_Tokenizer::isConnectorPunctuation(const std::string& token) const {
    return token == "-" || token == "_" || token == "/" || token == "'";
}
