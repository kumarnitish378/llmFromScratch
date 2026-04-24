#include "model.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>

namespace nks_llm {

// ============================================================
//  ModelConfig Implementation
// ============================================================

ModelConfig ModelConfig::get_large_model() {
    ModelConfig cfg;
    cfg.vocab_size = 2048;
    cfg.embedding_dim = 1024;
    cfg.num_layers = 24;
    cfg.num_heads = 16;
    cfg.ff_dim = 4096;
    cfg.max_seq_length = 2048;
    cfg.learning_rate = 1e-4f;
    return cfg;
}

ModelConfig ModelConfig::get_base_model() {
    ModelConfig cfg;
    cfg.vocab_size = 2048;
    cfg.embedding_dim = 768;
    cfg.num_layers = 12;
    cfg.num_heads = 12;
    cfg.ff_dim = 3072;
    cfg.max_seq_length = 2048;
    cfg.learning_rate = 5e-4f;
    return cfg;
}

ModelConfig ModelConfig::get_small_model() {
    ModelConfig cfg;
    cfg.vocab_size = 2048;
    cfg.embedding_dim = 256;
    cfg.num_layers = 6;
    cfg.num_heads = 8;
    cfg.ff_dim = 1024;
    cfg.max_seq_length = 512;
    cfg.learning_rate = 1e-3f;
    return cfg;
}

// ============================================================
//  LLMModel Implementation
// ============================================================

LLMModel::LLMModel(const ModelConfig& config)
    : config_(config),
      token_embedding_(config.vocab_size, config.embedding_dim),
      pos_encoding_(config.embedding_dim, config.max_seq_length),
      final_norm_(config.embedding_dim),
      lm_head_(config.embedding_dim, config.vocab_size) {
    
    // Create transformer layers
    for (size_t i = 0; i < config.num_layers; ++i) {
        transformer_layers_.emplace_back(
            config.embedding_dim,
            config.num_heads,
            config.ff_dim,
            config.dropout_prob
        );
    }
    
    // Calculate total parameters
    num_parameters_ = 0;
    
    // Embedding: vocab_size * embedding_dim
    num_parameters_ += config.vocab_size * config.embedding_dim;
    
    // Positional encoding: max_seq_length * embedding_dim (not trainable, but counted)
    num_parameters_ += config.max_seq_length * config.embedding_dim;
    
    // Each transformer layer:
    // - LayerNorm 1: 2 * embedding_dim (weight, bias)
    // - MultiHeadAttention:
    //   - Q, K, V, Out projections: 4 * (embedding_dim * embedding_dim + embedding_dim)
    // - LayerNorm 2: 2 * embedding_dim
    // - FeedForward:
    //   - Linear1: embedding_dim * ff_dim + ff_dim
    //   - Linear2: ff_dim * embedding_dim + embedding_dim
    
    size_t per_layer_params = 0;
    per_layer_params += 2 * config.embedding_dim;  // norm1 weight + bias
    per_layer_params += 4 * (config.embedding_dim * config.embedding_dim + config.embedding_dim);  // attention
    per_layer_params += 2 * config.embedding_dim;  // norm2 weight + bias
    per_layer_params += config.embedding_dim * config.ff_dim + config.ff_dim;  // ffn linear1
    per_layer_params += config.ff_dim * config.embedding_dim + config.embedding_dim;  // ffn linear2
    
    num_parameters_ += per_layer_params * config.num_layers;
    
    // Final norm: embedding_dim * 2 (weight + bias)
    num_parameters_ += 2 * config.embedding_dim;
    
    // LM head: embedding_dim * vocab_size + vocab_size
    num_parameters_ += config.embedding_dim * config.vocab_size + config.vocab_size;
    
    initialize_weights();
}

void LLMModel::initialize_weights() {
    // Initialize optimizer states for Adam
    m_states_.clear();
    v_states_.clear();
    
    // Create dummy m and v states for all learnable parameters
    for (size_t i = 0; i < config_.num_layers; ++i) {
        // For each transformer layer, create state tensors
        // Simplified: just allocate large tensors for now
        Tensor m({config_.embedding_dim * config_.embedding_dim});
        Tensor v({config_.embedding_dim * config_.embedding_dim});
        m.zeros_();
        v.zeros_();
        m_states_.push_back(m);
        v_states_.push_back(v);
    }
}

Tensor LLMModel::compute_causal_mask(size_t seq_length) const {
    // Create causal mask: upper triangular matrix of zeros (masked positions)
    // Shape: (seq_length, seq_length)
    Tensor mask({seq_length, seq_length});
    
    float* mask_ptr = mask.data();
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < seq_length; ++j) {
            if (j > i) {
                mask_ptr[i * seq_length + j] = 0.0f;  // Masked
            } else {
                mask_ptr[i * seq_length + j] = 1.0f;  // Not masked
            }
        }
    }
    
    return mask;
}

Tensor LLMModel::forward(const Tensor& input_ids) {
    // input_ids: (batch_size, seq_length)
    assert(input_ids.ndim() == 2);
    
    size_t batch_size = input_ids.shape()[0];
    size_t seq_length = input_ids.shape()[1];
    assert(seq_length <= config_.max_seq_length);
    
    // Token embeddings: (batch, seq_len, embed_dim)
    Tensor embeddings = token_embedding_.forward(input_ids);
    
    // Get positional encoding: (seq_len, embed_dim)
    Tensor pos_enc = pos_encoding_.forward(seq_length);
    
    // Add positional encoding (broadcasting across batch)
    // embeddings: (batch, seq_len, embed_dim), pos_enc: (seq_len, embed_dim)
    // We need to add these element-wise
    size_t embed_dim = config_.embedding_dim;
    float* emb_ptr = embeddings.data();
    const float* pos_ptr = pos_enc.data();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_length; ++s) {
            for (size_t d = 0; d < embed_dim; ++d) {
                emb_ptr[b * seq_length * embed_dim + s * embed_dim + d] += 
                    pos_ptr[s * embed_dim + d];
            }
        }
    }
    
    Tensor hidden_states = embeddings;
    
    // Create causal mask
    Tensor causal_mask = compute_causal_mask(seq_length);
    
    // Pass through transformer layers
    for (auto& layer : transformer_layers_) {
        hidden_states = layer.forward(hidden_states, &causal_mask);
    }
    
    // Apply final layer norm
    hidden_states = final_norm_.forward(hidden_states);
    
    // Project to vocabulary
    Tensor logits = compute_logits(hidden_states);  // (batch, seq_len, vocab_size)
    
    return logits;
}

Tensor LLMModel::compute_logits(const Tensor& hidden_states) {
    // hidden_states: (batch, seq_len, embed_dim)
    // output: (batch, seq_len, vocab_size)
    
    assert(hidden_states.ndim() == 3);
    
    // Apply LM head
    Tensor logits = lm_head_.forward(hidden_states);
    
    return logits;
}

float LLMModel::compute_cross_entropy_loss(const Tensor& logits, const Tensor& target_ids) {
    // logits: (batch_size, seq_length, vocab_size)
    // target_ids: (batch_size, seq_length)
    
    assert(logits.ndim() == 3);
    size_t batch_size = logits.shape()[0];
    size_t seq_length = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];
    
    float total_loss = 0.0f;
    size_t num_tokens = 0;
    
    const float* logits_ptr = logits.data();
    const float* target_ptr = target_ids.data();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            int target_id = static_cast<int>(target_ptr[b * seq_length + t]);
            if (target_id < 0 || target_id >= static_cast<int>(vocab_size)) {
                continue;  // Skip invalid targets
            }
            
            // Get logits for this position
            const float* pos_logits = logits_ptr + (b * seq_length + t) * vocab_size;
            
            // Find max logit for numerical stability
            float max_logit = pos_logits[0];
            for (size_t v = 0; v < vocab_size; ++v) {
                max_logit = std::max(max_logit, pos_logits[v]);
            }
            
            // Compute log-softmax
            float log_sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                log_sum_exp += std::exp(pos_logits[v] - max_logit);
            }
            log_sum_exp = std::log(log_sum_exp) + max_logit;
            
            // Compute cross-entropy loss for this token
            float loss = -(pos_logits[target_id] - log_sum_exp);
            total_loss += loss;
            num_tokens++;
        }
    }
    
    return num_tokens > 0 ? total_loss / static_cast<float>(num_tokens) : 0.0f;
}

LLMModel::TrainStep LLMModel::training_step(const Tensor& input_ids, const Tensor& target_ids) {
    // Forward pass
    Tensor logits = forward(input_ids);
    
    // Compute loss
    float loss = compute_cross_entropy_loss(logits, target_ids);
    
    // Compute perplexity
    float perplexity = std::exp(loss);
    
    TrainStep step;
    step.loss = loss;
    step.perplexity = perplexity;
    step.learning_rate = config_.learning_rate;
    
    return step;
}

std::vector<int> LLMModel::generate(const std::vector<int>& prompt, size_t max_new_tokens) {
    std::vector<int> sequence = prompt;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < max_new_tokens; ++i) {
        // Prepare input: last max_seq_length tokens
        size_t start_idx = sequence.size() > config_.max_seq_length 
                           ? sequence.size() - config_.max_seq_length 
                           : 0;
        std::vector<int> input_slice(sequence.begin() + start_idx, sequence.end());
        
        // Pad to full length if needed
        while (input_slice.size() < config_.max_seq_length) {
            input_slice.insert(input_slice.begin(), 0);  // Pad with zeros
        }
        
        // Create batch tensor (batch_size=1)
        Tensor input_tensor({1, static_cast<size_t>(input_slice.size())});
        for (size_t j = 0; j < input_slice.size(); ++j) {
            input_tensor[j] = static_cast<float>(input_slice[j]);
        }
        
        // Forward pass
        Tensor logits = forward(input_tensor);
        
        // Get logits for last position
        size_t last_pos = input_slice.size() - 1;
        const float* last_logits = logits.data() + last_pos * config_.vocab_size;
        
        // Apply temperature
        std::vector<float> adjusted_logits(config_.vocab_size);
        for (size_t v = 0; v < config_.vocab_size; ++v) {
            adjusted_logits[v] = last_logits[v] / config_.temperature;
        }
        
        // Apply softmax
        float max_logit = *std::max_element(adjusted_logits.begin(), adjusted_logits.end());
        float sum_exp = 0.0f;
        std::vector<float> probs(config_.vocab_size);
        for (size_t v = 0; v < config_.vocab_size; ++v) {
            probs[v] = std::exp(adjusted_logits[v] - max_logit);
            sum_exp += probs[v];
        }
        for (size_t v = 0; v < config_.vocab_size; ++v) {
            probs[v] /= sum_exp;
        }
        
        // Sample next token (greedy for now, could add top-k/nucleus sampling)
        int next_token = 0;
        float max_prob = probs[0];
        for (size_t v = 1; v < config_.vocab_size; ++v) {
            if (probs[v] > max_prob) {
                max_prob = probs[v];
                next_token = v;
            }
        }
        
        sequence.push_back(next_token);
    }
    
    return sequence;
}

bool LLMModel::save(const std::string& checkpoint_path) const {
    std::ofstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << checkpoint_path << std::endl;
        return false;
    }
    
    // Save config
    file.write(reinterpret_cast<const char*>(&config_.vocab_size), sizeof(config_.vocab_size));
    file.write(reinterpret_cast<const char*>(&config_.embedding_dim), sizeof(config_.embedding_dim));
    file.write(reinterpret_cast<const char*>(&config_.num_layers), sizeof(config_.num_layers));
    file.write(reinterpret_cast<const char*>(&config_.num_heads), sizeof(config_.num_heads));
    
    // Save embeddings
    token_embedding_.save(checkpoint_path + ".embedding");
    
    // Save transformer layers
    for (size_t i = 0; i < transformer_layers_.size(); ++i) {
        std::string layer_path = checkpoint_path + ".layer_" + std::to_string(i);
        transformer_layers_[i].save(layer_path);
    }
    
    // Save output layer
    final_norm_.save(checkpoint_path + ".norm");
    lm_head_.save(checkpoint_path + ".lm_head");
    
    std::cout << "Model saved to: " << checkpoint_path << std::endl;
    return file.good();
}

bool LLMModel::load(const std::string& checkpoint_path) {
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << checkpoint_path << std::endl;
        return false;
    }
    
    size_t saved_vocab, saved_embed, saved_layers, saved_heads;
    file.read(reinterpret_cast<char*>(&saved_vocab), sizeof(saved_vocab));
    file.read(reinterpret_cast<char*>(&saved_embed), sizeof(saved_embed));
    file.read(reinterpret_cast<char*>(&saved_layers), sizeof(saved_layers));
    file.read(reinterpret_cast<char*>(&saved_heads), sizeof(saved_heads));
    
    if (!file.good()) return false;
    
    // Load embeddings
    token_embedding_.load(checkpoint_path + ".embedding");
    
    // Load transformer layers
    for (size_t i = 0; i < transformer_layers_.size(); ++i) {
        std::string layer_path = checkpoint_path + ".layer_" + std::to_string(i);
        transformer_layers_[i].load(layer_path);
    }
    
    // Load output layer
    final_norm_.load(checkpoint_path + ".norm");
    lm_head_.load(checkpoint_path + ".lm_head");
    
    std::cout << "Model loaded from: " << checkpoint_path << std::endl;
    return true;
}

float LLMModel::compute_gradient_norm() const {
    // Placeholder: return dummy value
    return 0.0f;
}

void LLMModel::clip_gradients(float max_norm) {
    // Placeholder: implement gradient clipping
}

void LLMModel::zero_gradients() {
    // Placeholder: zero out gradients
}

void LLMModel::optimizer_step(float learning_rate) {
    // Placeholder: implement Adam optimizer step
    optimizer_step_count_++;
}

}  // namespace nks_llm
