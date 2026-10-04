#include "app_runner.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "libraries/NKS_Tokenizer/NKS_Tokenizer.h"
#include "libraries/NKS_LLM/NKS_LLM.h"

namespace {
std::size_t readSize(const std::string& prompt, std::size_t fallback) {
    std::cout << prompt << " (default=" << fallback << "): ";
    std::string value;
    if (!std::getline(std::cin, value) || value.empty()) return fallback;
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (...) {
        std::cout << "Invalid value; using " << fallback << ".\n";
        return fallback;
    }
}
}

int runTransformerCorpusTrainingExample() {
    using namespace nks_llm;
    namespace fs = std::filesystem;

    const std::string corpusPath = "Data/clean_training_corpus.txt";
    const std::string tokenizerPath = "Metadata/bpe_model_processed.bin";
    const std::string checkpointPath = "Metadata/transformer_corpus_checkpoint.bin";

    std::cout << "\n========================================\n"
              << "   Transformer Corpus Training (C++/CUDA)\n"
              << "========================================\n";

    if (!fs::exists(corpusPath)) {
        std::cerr << "Training corpus not found: " << corpusPath << "\n"
                  << "Set NKS_BPE_TRAINING_PATH in your environment or place the corpus at the path above.\n";
        return 1;
    }

    NKS_Tokenizer tokenizer;
    if (!tokenizer.loadModel(tokenizerPath)) {
        std::cout << "Tokenizer model not found; building BPE vocabulary from corpus...\n";
        if (!tokenizer.loadVocabulary(corpusPath)) {
            std::cerr << "Could not load/train tokenizer from " << corpusPath << "\n";
            return 1;
        }
        fs::create_directories(fs::path(tokenizerPath).parent_path());
        if (!tokenizer.saveModel(tokenizerPath)) {
            std::cerr << "Warning: could not save tokenizer model. Continuing with in-memory tokenizer.\n";
        }
    }

    std::vector<int> tokens;
    tokens.reserve(200000);
    const std::size_t maxLines = readSize("Maximum corpus lines to read (0 = no limit)", 100000);
    const std::size_t maxTokens = readSize("Maximum tokens to keep in RAM", 1000000);
    std::ifstream corpus(corpusPath);
    std::string line;
    std::size_t linesRead = 0;
    while (std::getline(corpus, line)) {
        if (line.empty()) continue;
        const std::vector<int> encoded = tokenizer.encode(line);
        for (int id : encoded) {
            if (id >= 0) tokens.push_back(id);
            if (maxTokens > 0 && tokens.size() >= maxTokens) break;
        }
        ++linesRead;
        if (linesRead % 1000 == 0) {
            std::cout << "\rRead lines: " << linesRead << " | tokens: " << tokens.size() << std::flush;
        }
        if ((maxLines > 0 && linesRead >= maxLines) ||
            (maxTokens > 0 && tokens.size() >= maxTokens)) break;
    }
    std::cout << "\nLoaded " << linesRead << " lines and " << tokens.size() << " tokens.\n";
    if (tokens.size() < 3) {
        std::cerr << "Not enough tokens to train. Check the corpus and tokenizer.\n";
        return 1;
    }

    const std::size_t epochs = readSize("Training epochs", 1);
    const std::size_t seqLen = readSize("Sequence length (keep small for first run)", 32);
    if (epochs == 0 || seqLen < 2 || seqLen > 128) {
        std::cerr << "Use epochs >= 1 and sequence length between 2 and 128.\n";
        return 1;
    }

    ModelConfig config;
    config.vocab_size = std::max<std::size_t>(tokenizer.vocabularySize(), 256);
    for (int id : tokens) config.vocab_size = std::max(config.vocab_size, static_cast<std::size_t>(id) + 1);
    config.max_seq_length = seqLen;
    config.embedding_dim = 128;
    config.num_layers = 2;
    config.num_heads = 4;
    config.ff_dim = 512;
    config.batch_size = 1;
    config.num_epochs = epochs;
    config.learning_rate = 1e-4f;
    config.dropout_prob = 0.0f;

    std::cout << "Model: vocab=" << config.vocab_size
              << ", dim=" << config.embedding_dim
              << ", layers=" << config.num_layers
              << ", heads=" << config.num_heads
              << ", sequence=" << seqLen << "\n";
    std::cout << "Backend: " << gpu_backend::backend_name();
    if (!gpu_backend::is_available()) std::cout << " (CUDA unavailable; backend may use CPU)";
    std::cout << "\n";

    LLMModel model(config);
    const std::size_t samplesPerEpoch = (tokens.size() - 1) / seqLen;
    if (samplesPerEpoch == 0) {
        std::cerr << "Corpus does not contain enough tokens for one sequence.\n";
        return 1;
    }

    float firstLoss = 0.0f;
    float lastLoss = 0.0f;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        double lossSum = 0.0;
        std::size_t steps = 0;
        for (std::size_t offset = 0; offset + seqLen < tokens.size(); offset += seqLen) {
            Tensor input({1, seqLen});
            Tensor target({1, seqLen});
            for (std::size_t t = 0; t < seqLen; ++t) {
                input[t] = static_cast<float>(tokens[offset + t] % config.vocab_size);
                target[t] = static_cast<float>(tokens[offset + t + 1] % config.vocab_size);
            }
            const auto step = model.training_step(input, target);
            if (steps == 0 && epoch == 0) firstLoss = step.loss;
            lastLoss = step.loss;
            lossSum += step.loss;
            ++steps;
            if (steps % 10 == 0 || steps == samplesPerEpoch) {
                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                          << " | step " << steps << "/" << samplesPerEpoch
                          << " | loss " << step.loss
                          << " | perplexity " << step.perplexity << std::flush;
            }
        }
        std::cout << "\nEpoch " << (epoch + 1) << " average loss: "
                  << (steps ? lossSum / static_cast<double>(steps) : 0.0) << "\n";
    }

    fs::create_directories(fs::path(checkpointPath).parent_path());
    if (!model.save(checkpointPath)) {
        std::cerr << "Training ran, but checkpoint save failed: " << checkpointPath << "\n";
        return 1;
    }

    std::cout << "\nTraining finished. Initial loss: " << firstLoss
              << " | final loss: " << lastLoss
              << "\nCheckpoint: " << checkpointPath
              << "\nNote: this is a small prototype configuration; inspect loss and validate generation before scaling up.\n";
    return 0;
}
