#include "NKS_SentencePieceTokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {
std::string toLowerCopy(const std::string& value) {
    std::string out = value;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool isCsvPath(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    return toLowerCopy(path.substr(dot)) == ".csv";
}

std::vector<std::string> parseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
            continue;
        }

        if (ch == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
            continue;
        }
        field.push_back(ch);
    }

    fields.push_back(field);
    return fields;
}

std::vector<std::string> loadCorpusRows(const std::string& path, std::size_t maxRows) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return {};
    }

    std::vector<std::string> rows;
    if (maxRows > 0) {
        rows.reserve(maxRows);
    }

    if (!isCsvPath(path)) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            rows.push_back(line);
            if (maxRows > 0 && rows.size() >= maxRows) {
                break;
            }
        }
        return rows;
    }

    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        return rows;
    }
    const std::vector<std::string> header = parseCsvRow(headerLine);
    std::size_t textColumn = 0;
    bool foundTextColumn = false;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (toLowerCopy(header[i]) == "text") {
            textColumn = i;
            foundTextColumn = true;
            break;
        }
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = parseCsvRow(line);
        std::string text;
        if (foundTextColumn && textColumn < fields.size()) {
            text = fields[textColumn];
        } else if (!fields.empty()) {
            text = fields[0];
        }

        if (text.empty()) {
            continue;
        }
        rows.push_back(text);
        if (maxRows > 0 && rows.size() >= maxRows) {
            break;
        }
    }

    return rows;
}
} // namespace

NKS_SentencePieceTokenizer::NKS_SentencePieceTokenizer() {
    resetVocabulary();
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setTargetVocabSize(std::size_t size) {
    if (size >= kMinTargetVocab) {
        config_.targetVocabSize = size;
    }
    return *this;
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setMaxPieceChars(std::size_t size) {
    if (size >= kMinPieceChars) {
        config_.maxPieceChars = size;
    }
    return *this;
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setTrainingLineLimit(std::size_t limit) {
    if (limit >= kMinLineLimit) {
        config_.trainingLineLimit = limit;
    }
    return *this;
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setLowercase(bool enabled) {
    config_.lowercase = enabled;
    return *this;
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setSplitCamelCase(bool enabled) {
    config_.splitCamelCase = enabled;
    return *this;
}

NKS_SentencePieceTokenizer& NKS_SentencePieceTokenizer::setTrainingConfig(const TrainingConfig& config) {
    config_ = config;
    if (config_.targetVocabSize < kMinTargetVocab) {
        config_.targetVocabSize = kMinTargetVocab;
    }
    if (config_.maxPieceChars < kMinPieceChars) {
        config_.maxPieceChars = kMinPieceChars;
    }
    if (config_.trainingLineLimit < kMinLineLimit) {
        config_.trainingLineLimit = kMinLineLimit;
    }
    return *this;
}

bool NKS_SentencePieceTokenizer::trainFromFile(const std::string& corpusPath) {
    const std::vector<std::string> corpusRows = loadCorpusRows(corpusPath, config_.trainingLineLimit);
    if (corpusRows.empty()) {
        return false;
    }

    std::vector<std::string> corpusLines;
    corpusLines.reserve(config_.trainingLineLimit);

    for (const std::string& row : corpusRows) {
        const std::string normalized = normalizeInputForTraining(row);
        if (normalized.empty()) {
            continue;
        }
        corpusLines.push_back(normalized);
    }

    if (corpusLines.empty()) {
        return false;
    }

    resetVocabulary();
    buildSeedCharacters(corpusLines);
    addFrequentSubstringCandidates(corpusLines);
    return true;
}

std::vector<std::string> NKS_SentencePieceTokenizer::encode(const std::string& text) const {
    const std::string normalized = normalizeInputForEncoding(text);
    const std::vector<Utf8Char> chars = splitUtf8Chars(normalized);

    if (chars.empty()) {
        return {};
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> bestCost(chars.size() + 1, inf);
    std::vector<int> prevIndex(chars.size() + 1, -1);
    std::vector<std::string> prevPiece(chars.size() + 1);
    bestCost[0] = 0.0;

    for (std::size_t i = 0; i < chars.size(); ++i) {
        if (bestCost[i] == inf) {
            continue;
        }

        std::string candidate;
        candidate.reserve(config_.maxPieceChars * kUtf8ByteReserveFactor);

        const std::size_t maxLen = std::min(config_.maxPieceChars, chars.size() - i);
        for (std::size_t len = 1; len <= maxLen; ++len) {
            candidate.append(chars[i + len - 1].bytes);
            const auto it = pieceLogProb_.find(candidate);
            if (it == pieceLogProb_.end()) {
                continue;
            }

            const std::size_t j = i + len;
            const double cost = bestCost[i] - it->second;
            if (cost < bestCost[j]) {
                bestCost[j] = cost;
                prevIndex[j] = static_cast<int>(i);
                prevPiece[j] = candidate;
            }
        }

        if (prevIndex[i + 1] == -1) {
            const std::string& oneChar = chars[i].bytes;
            const double fallbackCost = bestCost[i] + kFallbackCost;
            if (fallbackCost < bestCost[i + 1]) {
                bestCost[i + 1] = fallbackCost;
                prevIndex[i + 1] = static_cast<int>(i);
                prevPiece[i + 1] = oneChar;
            }
        }
    }

    std::vector<std::string> pieces;
    int cursor = static_cast<int>(chars.size());
    while (cursor > 0 && prevIndex[cursor] >= 0) {
        pieces.push_back(prevPiece[cursor]);
        cursor = prevIndex[cursor];
    }

    std::reverse(pieces.begin(), pieces.end());
    return pieces;
}

std::vector<int> NKS_SentencePieceTokenizer::encodeToIds(const std::string& text) const {
    const std::vector<std::string> pieces = encode(text);
    std::vector<int> ids;
    ids.reserve(pieces.size());

    for (const std::string& piece : pieces) {
        const auto it = pieceToId_.find(piece);
        ids.push_back(it != pieceToId_.end() ? it->second : unknownId_);
    }
    return ids;
}

std::string NKS_SentencePieceTokenizer::decode(const std::vector<std::string>& pieces) const {
    std::string out;
    for (const std::string& p : pieces) {
        out.append(p);
    }

    const std::string marker = kSpaceMarker;
    std::size_t pos = 0;
    while ((pos = out.find(marker, pos)) != std::string::npos) {
        out.replace(pos, marker.size(), " ");
        pos += 1;
    }

    return trimAsciiSpaces(out);
}

std::string NKS_SentencePieceTokenizer::decodeFromIds(const std::vector<int>& ids) const {
    std::vector<std::string> pieces;
    pieces.reserve(ids.size());

    for (int id : ids) {
        if (id >= 0 && static_cast<std::size_t>(id) < idToPiece_.size()) {
            pieces.push_back(idToPiece_[id]);
        } else {
            pieces.push_back("<unk>");
        }
    }
    return decode(pieces);
}

std::size_t NKS_SentencePieceTokenizer::vocabularySize() const {
    return idToPiece_.size();
}

std::vector<NKS_SentencePieceTokenizer::Utf8Char> NKS_SentencePieceTokenizer::splitUtf8Chars(const std::string& text) {
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

char NKS_SentencePieceTokenizer::toLowerAscii(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool NKS_SentencePieceTokenizer::isAsciiLower(uint32_t cp) {
    return cp >= 'a' && cp <= 'z';
}

bool NKS_SentencePieceTokenizer::isAsciiUpper(uint32_t cp) {
    return cp >= 'A' && cp <= 'Z';
}

bool NKS_SentencePieceTokenizer::isUnicodeWhitespace(uint32_t cp) {
    if (cp <= 127) {
        return std::isspace(static_cast<unsigned char>(cp)) != 0;
    }
    return cp == 0x00A0u || cp == 0x1680u || cp == 0x2007u || cp == 0x202Fu || cp == 0x3000u;
}

std::string NKS_SentencePieceTokenizer::trimAsciiSpaces(const std::string& s) {
    const std::size_t first = s.find_first_not_of(' ');
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(' ');
    return s.substr(first, last - first + 1);
}

std::string NKS_SentencePieceTokenizer::normalizeInputForTraining(const std::string& text) const {
    return normalizeInputForEncoding(text);
}

std::string NKS_SentencePieceTokenizer::normalizeInputForEncoding(const std::string& text) const {
    const std::vector<Utf8Char> chars = splitUtf8Chars(text);
    std::string normalized;
    normalized.reserve(text.size() + kEncodingReservePadding);

    bool needBoundary = true;
    bool hasPrevNonSpace = false;
    uint32_t prevCp = 0;
    for (const Utf8Char& ch : chars) {
        if (isUnicodeWhitespace(ch.codepoint)) {
            needBoundary = true;
            hasPrevNonSpace = false;
            continue;
        }

        if (config_.splitCamelCase && hasPrevNonSpace && isAsciiLower(prevCp) && isAsciiUpper(ch.codepoint)) {
            needBoundary = true;
        }

        if (needBoundary) {
            normalized.append(kSpaceMarker);
            needBoundary = false;
        }

        if (config_.lowercase && ch.codepoint <= 127) {
            normalized.push_back(toLowerAscii(ch.bytes[0]));
        } else {
            normalized.append(ch.bytes);
        }

        prevCp = ch.codepoint;
        hasPrevNonSpace = true;
    }

    return normalized;
}

void NKS_SentencePieceTokenizer::resetVocabulary() {
    pieceToId_.clear();
    idToPiece_.clear();
    pieceLogProb_.clear();

    idToPiece_.push_back("<unk>");
    pieceToId_["<unk>"] = 0;
    pieceLogProb_["<unk>"] = kUnknownLogProb;
    unknownId_ = 0;
}

void NKS_SentencePieceTokenizer::buildSeedCharacters(const std::vector<std::string>& corpusLines) {
    std::unordered_map<std::string, std::size_t> counts;

    for (const std::string& line : corpusLines) {
        for (const Utf8Char& ch : splitUtf8Chars(line)) {
            ++counts[ch.bytes];
        }
    }

    std::size_t total = 0;
    for (const auto& it : counts) {
        total += it.second;
    }
    if (total == 0) {
        total = 1;
    }

    for (const auto& it : counts) {
        const int id = static_cast<int>(idToPiece_.size());
        pieceToId_[it.first] = id;
        idToPiece_.push_back(it.first);
        pieceLogProb_[it.first] = std::log(static_cast<double>(it.second) / static_cast<double>(total));
    }
}

void NKS_SentencePieceTokenizer::addFrequentSubstringCandidates(const std::vector<std::string>& corpusLines) {
    std::unordered_map<std::string, std::size_t> ngramCount;

    for (const std::string& line : corpusLines) {
        const std::vector<Utf8Char> chars = splitUtf8Chars(line);
        if (chars.empty()) {
            continue;
        }

        const std::size_t maxLen = std::min(config_.maxPieceChars, chars.size());
        for (std::size_t i = 0; i < chars.size(); ++i) {
            std::string piece;
            piece.reserve(config_.maxPieceChars * kUtf8ByteReserveFactor);

            for (std::size_t len = 1; len <= maxLen && i + len <= chars.size(); ++len) {
                piece.append(chars[i + len - 1].bytes);
                if (len == 1) {
                    continue;
                }
                ++ngramCount[piece];
            }
        }
    }

    std::vector<std::pair<std::string, std::size_t>> candidates;
    candidates.reserve(ngramCount.size());
    std::size_t total = 0;
    for (const auto& it : ngramCount) {
        if (it.second < kMinNgramFrequency) {
            continue;
        }
        candidates.push_back(it);
        total += it.second;
    }
    if (total == 0) {
        total = 1;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first.size() > b.first.size();
    });

    for (const auto& it : candidates) {
        if (idToPiece_.size() >= config_.targetVocabSize) {
            break;
        }
        if (pieceToId_.find(it.first) != pieceToId_.end()) {
            continue;
        }

        const int id = static_cast<int>(idToPiece_.size());
        pieceToId_[it.first] = id;
        idToPiece_.push_back(it.first);
        pieceLogProb_[it.first] = std::log(static_cast<double>(it.second) / static_cast<double>(total));
    }
}
