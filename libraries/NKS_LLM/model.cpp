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
    
    // Initialize optimizer and scheduler
    Adam::Config adam_config;
    adam_config.learning_rate = config.learning_rate;
    adam_config.beta1 = config.adam_beta1;
    adam_config.beta2 = config.adam_beta2;
    adam_config.epsilon = config.adam_eps;
    adam_config.weight_decay = config.weight_decay;
    adam_config.gradient_clip = config.gradient_clip;
    
    optimizer_ = std::make_unique<Adam>(adam_config);
    lr_scheduler_ = std::make_unique<LRScheduler>(config.learning_rate);
    
    initialize_weights();
}

void LLMModel::initialize_weights() {
    // Initialize optimizer states for Adam (kept for compatibility)
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
    assert(input_ids.ndim() == 2);
    assert(target_ids.ndim() == 2);
    
    size_t batch_size = input_ids.shape()[0];
    size_t seq_length = input_ids.shape()[1];
    size_t embed_dim = config_.embedding_dim;
    size_t vocab_size = config_.vocab_size;
    
    // --------------------------------------------------------
    // 1. Forward Pass (storing activations needed for backprop)
    // --------------------------------------------------------
    Tensor embeddings = token_embedding_.forward(input_ids);
    Tensor pos_enc = pos_encoding_.forward(seq_length);
    
    // Add positional encoding
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
    
    Tensor causal_mask = compute_causal_mask(seq_length);
    
    struct LayerCache {
        Tensor x_in;
        Tensor norm1_out;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor attn_weights;
        Tensor attn_out;
        Tensor x_after_attn;
        Tensor norm2_out;
        Tensor h1;
        Tensor a1;
        Tensor ffn_out;
    };
    
    std::vector<LayerCache> caches(transformer_layers_.size());
    Tensor hidden_states = embeddings;
    
    for (size_t i = 0; i < transformer_layers_.size(); ++i) {
        auto& layer = transformer_layers_[i];
        auto& cache = caches[i];
        
        cache.x_in = hidden_states;
        cache.norm1_out = layer.norm1().forward(cache.x_in);
        
        // Multi-head attention forward with cache
        cache.q = layer.attn().q_linear().forward(cache.norm1_out);
        cache.k = layer.attn().k_linear().forward(cache.norm1_out);
        cache.v = layer.attn().v_linear().forward(cache.norm1_out);
        
        // Attention scores
        Tensor k_t = Tensor::transpose(cache.k, 1, 2);
        Tensor scores = Tensor::matmul(cache.q, k_t);
        float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim));
        scores *= scale;
        
        // Apply causal mask
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t r = 0; r < seq_length; ++r) {
                for (size_t c = 0; c < seq_length; ++c) {
                    if (causal_mask[r * seq_length + c] == 0.0f) {
                        scores[b * seq_length * seq_length + r * seq_length + c] = -1e4f;
                    }
                }
            }
        }
        
        cache.attn_weights = Tensor::softmax(scores, -1);
        Tensor attn_proj_in = Tensor::matmul(cache.attn_weights, cache.v);
        cache.attn_out = layer.attn().out_linear().forward(attn_proj_in);
        
        cache.x_after_attn = cache.x_in + cache.attn_out;
        cache.norm2_out = layer.norm2().forward(cache.x_after_attn);
        
        // FFN forward with cache
        cache.h1 = layer.ffn().linear1().forward(cache.norm2_out);
        cache.a1 = Tensor::gelu(cache.h1);
        cache.ffn_out = layer.ffn().linear2().forward(cache.a1);
        
        hidden_states = cache.x_after_attn + cache.ffn_out;
    }
    
    Tensor final_norm_out = final_norm_.forward(hidden_states);
    Tensor logits = lm_head_.forward(final_norm_out);
    
    // --------------------------------------------------------
    // 2. Cross-Entropy Loss & Exact Softmax Gradients (dLogits)
    // --------------------------------------------------------
    Tensor dlogits({batch_size, seq_length, vocab_size}, true);
    dlogits.zeros_();
    
    float total_loss = 0.0f;
    size_t num_valid_tokens = 0;
    const float* logits_ptr = logits.data();
    const float* target_ptr = target_ids.data();
    float* dlogits_ptr = dlogits.data();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            int target_id = static_cast<int>(target_ptr[b * seq_length + t]);
            if (target_id >= 0 && target_id < static_cast<int>(vocab_size)) {
                num_valid_tokens++;
            }
        }
    }
    
    if (num_valid_tokens == 0) num_valid_tokens = 1;
    float norm_factor = 1.0f / static_cast<float>(num_valid_tokens);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            int target_id = static_cast<int>(target_ptr[b * seq_length + t]);
            if (target_id < 0 || target_id >= static_cast<int>(vocab_size)) {
                continue;
            }
            
            const float* pos_logits = logits_ptr + (b * seq_length + t) * vocab_size;
            float* pos_dlogits = dlogits_ptr + (b * seq_length + t) * vocab_size;
            
            float max_l = pos_logits[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (pos_logits[v] > max_l) max_l = pos_logits[v];
            }
            
            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                sum_exp += std::exp(pos_logits[v] - max_l);
            }
            float inv_sum = 1.0f / std::max(sum_exp, 1e-12f);
            
            float loss = -( (pos_logits[target_id] - max_l) - std::log(std::max(sum_exp, 1e-12f)) );
            total_loss += loss;
            
            for (size_t v = 0; v < vocab_size; ++v) {
                float prob = std::exp(pos_logits[v] - max_l) * inv_sum;
                if (static_cast<int>(v) == target_id) {
                    pos_dlogits[v] = (prob - 1.0f) * norm_factor;
                } else {
                    pos_dlogits[v] = prob * norm_factor;
                }
            }
        }
    }
    
    float avg_loss = total_loss * norm_factor;
    float perplexity = std::exp(std::min(avg_loss, 20.0f));
    
    // --------------------------------------------------------
    // 3. Backward Pass & Parameter Updates with Adam
    // --------------------------------------------------------
    size_t M = batch_size * seq_length;
    
    // Reshape dlogits to (M, vocab_size)
    Tensor dlogits_2d({M, vocab_size});
    std::memcpy(dlogits_2d.data(), dlogits.data(), dlogits.elem_count() * sizeof(float));
    
    // Reshape final_norm_out to (M, embed_dim)
    Tensor final_norm_2d({M, embed_dim});
    std::memcpy(final_norm_2d.data(), final_norm_out.data(), final_norm_out.elem_count() * sizeof(float));
    
    // dW_head = dlogits_2d.T @ final_norm_2d: shape (vocab_size, embed_dim)
    Tensor dW_head = Tensor::matmul(Tensor::transpose(dlogits_2d, 0, 1), final_norm_2d);
    
    // db_head = sum over rows of dlogits_2d: shape (vocab_size)
    Tensor db_head({vocab_size}, true);
    db_head.zeros_();
    for (size_t m = 0; m < M; ++m) {
        for (size_t v = 0; v < vocab_size; ++v) {
            db_head[v] += dlogits_2d[m * vocab_size + v];
        }
    }
    
    // dX_final = dlogits_2d @ lm_head_.weight(): shape (M, embed_dim)
    Tensor dX_final_2d = Tensor::matmul(dlogits_2d, lm_head_.weight());
    
    // Update lm_head_
    optimizer_->update(lm_head_.weight(), dW_head);
    optimizer_->update(lm_head_.bias(), db_head);
    
    // Backprop through final_norm_
    Tensor d_gamma({embed_dim}, true);
    d_gamma.zeros_();
    for (size_t m = 0; m < M; ++m) {
        for (size_t d = 0; d < embed_dim; ++d) {
            d_gamma[d] += dX_final_2d[m * embed_dim + d] * final_norm_2d[m * embed_dim + d];
        }
    }
    optimizer_->update(final_norm_.weight(), d_gamma);
    
    // Current gradient tensor flowing back: shape (M, embed_dim)
    Tensor dX_cur_2d({M, embed_dim});
    const float* gamma_ptr = final_norm_.weight().data();
    for (size_t m = 0; m < M; ++m) {
        for (size_t d = 0; d < embed_dim; ++d) {
            dX_cur_2d[m * embed_dim + d] = dX_final_2d[m * embed_dim + d] * gamma_ptr[d];
        }
    }
    
    auto gelu_grad = [](float x, float g) -> float {
        const float k = 1.702f;
        float s = 1.0f / (1.0f + std::exp(-k * x));
        float dg = s + k * x * s * (1.0f - s);
        return g * dg;
    };
    
    // Backprop through transformer layers in reverse
    for (int l = static_cast<int>(transformer_layers_.size()) - 1; l >= 0; --l) {
        auto& layer = transformer_layers_[static_cast<size_t>(l)];
        const auto& cache = caches[static_cast<size_t>(l)];
        
        Tensor dO_ffn = dX_cur_2d;
        Tensor dX_mid = dX_cur_2d;
        
        // --- FFN Linear2 ---
        size_t ff_dim = config_.ff_dim;
        Tensor a1_2d({M, ff_dim});
        std::memcpy(a1_2d.data(), cache.a1.data(), cache.a1.elem_count() * sizeof(float));
        
        Tensor dW2 = Tensor::matmul(Tensor::transpose(dO_ffn, 0, 1), a1_2d);
        Tensor db2({embed_dim}, true);
        db2.zeros_();
        for (size_t m = 0; m < M; ++m) {
            for (size_t d = 0; d < embed_dim; ++d) {
                db2[d] += dO_ffn[m * embed_dim + d];
            }
        }
        
        Tensor dA1 = Tensor::matmul(dO_ffn, layer.ffn().linear2().weight());
        optimizer_->update(layer.ffn().linear2().weight(), dW2);
        optimizer_->update(layer.ffn().linear2().bias(), db2);
        
        // --- FFN GELU backward ---
        Tensor dH1({M, ff_dim});
        const float* h1_ptr = cache.h1.data();
        for (size_t i = 0; i < M * ff_dim; ++i) {
            dH1[i] = gelu_grad(h1_ptr[i], dA1[i]);
        }
        
        // --- FFN Linear1 ---
        Tensor norm2_2d({M, embed_dim});
        std::memcpy(norm2_2d.data(), cache.norm2_out.data(), cache.norm2_out.elem_count() * sizeof(float));
        
        Tensor dW1 = Tensor::matmul(Tensor::transpose(dH1, 0, 1), norm2_2d);
        Tensor db1({ff_dim}, true);
        db1.zeros_();
        for (size_t m = 0; m < M; ++m) {
            for (size_t f = 0; f < ff_dim; ++f) {
                db1[f] += dH1[m * ff_dim + f];
            }
        }
        
        Tensor dNorm2 = Tensor::matmul(dH1, layer.ffn().linear1().weight());
        optimizer_->update(layer.ffn().linear1().weight(), dW1);
        optimizer_->update(layer.ffn().linear1().bias(), db1);
        
        for (size_t i = 0; i < M * embed_dim; ++i) {
            dX_mid[i] += dNorm2[i];
        }
        
        // --- Attention Backward ---
        Tensor dO_attn = dX_mid;
        Tensor dX_in = dX_mid;
        
        Tensor attn_proj_in = Tensor::matmul(cache.attn_weights, cache.v);
        Tensor attn_proj_2d({M, embed_dim});
        std::memcpy(attn_proj_2d.data(), attn_proj_in.data(), attn_proj_in.elem_count() * sizeof(float));
        
        Tensor dW_out = Tensor::matmul(Tensor::transpose(dO_attn, 0, 1), attn_proj_2d);
        Tensor db_out({embed_dim}, true);
        db_out.zeros_();
        for (size_t m = 0; m < M; ++m) {
            for (size_t d = 0; d < embed_dim; ++d) {
                db_out[d] += dO_attn[m * embed_dim + d];
            }
        }
        
        Tensor d_attn_proj_2d = Tensor::matmul(dO_attn, layer.attn().out_linear().weight());
        optimizer_->update(layer.attn().out_linear().weight(), dW_out);
        optimizer_->update(layer.attn().out_linear().bias(), db_out);
        
        Tensor d_attn_proj_3d({batch_size, seq_length, embed_dim});
        std::memcpy(d_attn_proj_3d.data(), d_attn_proj_2d.data(), d_attn_proj_2d.elem_count() * sizeof(float));
        
        Tensor attn_w_t = Tensor::transpose(cache.attn_weights, 1, 2);
        Tensor dV = Tensor::matmul(attn_w_t, d_attn_proj_3d);
        
        Tensor v_t = Tensor::transpose(cache.v, 1, 2);
        Tensor d_attn_w = Tensor::matmul(d_attn_proj_3d, v_t);
        
        Tensor dScores({batch_size, seq_length, seq_length}, true);
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(embed_dim));
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t r = 0; r < seq_length; ++r) {
                float row_dot = 0.0f;
                for (size_t c = 0; c < seq_length; ++c) {
                    size_t idx = b * seq_length * seq_length + r * seq_length + c;
                    row_dot += d_attn_w[idx] * cache.attn_weights[idx];
                }
                for (size_t c = 0; c < seq_length; ++c) {
                    size_t idx = b * seq_length * seq_length + r * seq_length + c;
                    dScores[idx] = cache.attn_weights[idx] * (d_attn_w[idx] - row_dot) * attn_scale;
                }
            }
        }
        
        Tensor dQ = Tensor::matmul(dScores, cache.k);
        Tensor dScores_t = Tensor::transpose(dScores, 1, 2);
        Tensor dK = Tensor::matmul(dScores_t, cache.q);
        
        Tensor dQ_2d({M, embed_dim}), dK_2d({M, embed_dim}), dV_2d({M, embed_dim});
        std::memcpy(dQ_2d.data(), dQ.data(), dQ.elem_count() * sizeof(float));
        std::memcpy(dK_2d.data(), dK.data(), dK.elem_count() * sizeof(float));
        std::memcpy(dV_2d.data(), dV.data(), dV.elem_count() * sizeof(float));
        
        Tensor norm1_2d({M, embed_dim});
        std::memcpy(norm1_2d.data(), cache.norm1_out.data(), cache.norm1_out.elem_count() * sizeof(float));
        
        Tensor dWq = Tensor::matmul(Tensor::transpose(dQ_2d, 0, 1), norm1_2d);
        Tensor dWk = Tensor::matmul(Tensor::transpose(dK_2d, 0, 1), norm1_2d);
        Tensor dWv = Tensor::matmul(Tensor::transpose(dV_2d, 0, 1), norm1_2d);
        
        Tensor dbq({embed_dim}, true), dbk({embed_dim}, true), dbv({embed_dim}, true);
        dbq.zeros_(); dbk.zeros_(); dbv.zeros_();
        for (size_t m = 0; m < M; ++m) {
            for (size_t d = 0; d < embed_dim; ++d) {
                dbq[d] += dQ_2d[m * embed_dim + d];
                dbk[d] += dK_2d[m * embed_dim + d];
                dbv[d] += dV_2d[m * embed_dim + d];
            }
        }
        
        optimizer_->update(layer.attn().q_linear().weight(), dWq);
        optimizer_->update(layer.attn().q_linear().bias(), dbq);
        optimizer_->update(layer.attn().k_linear().weight(), dWk);
        optimizer_->update(layer.attn().k_linear().bias(), dbk);
        optimizer_->update(layer.attn().v_linear().weight(), dWv);
        optimizer_->update(layer.attn().v_linear().bias(), dbv);
        
        Tensor dNorm1 = Tensor::matmul(dQ_2d, layer.attn().q_linear().weight()) +
                        Tensor::matmul(dK_2d, layer.attn().k_linear().weight()) +
                        Tensor::matmul(dV_2d, layer.attn().v_linear().weight());
        
        for (size_t i = 0; i < M * embed_dim; ++i) {
            dX_in[i] += dNorm1[i];
        }
        
        dX_cur_2d = dX_in;
    }
    
    // --------------------------------------------------------
    // 4. Token Embeddings Backward
    // --------------------------------------------------------
    Tensor d_emb({vocab_size, embed_dim}, true);
    d_emb.zeros_();
    float* d_emb_ptr = d_emb.data();
    const float* dX_in_ptr = dX_cur_2d.data();
    const float* inp_ptr = input_ids.data();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_length; ++s) {
            int token_id = static_cast<int>(inp_ptr[b * seq_length + s]);
            if (token_id >= 0 && token_id < static_cast<int>(vocab_size)) {
                for (size_t d = 0; d < embed_dim; ++d) {
                    d_emb_ptr[token_id * embed_dim + d] += 
                        dX_in_ptr[(b * seq_length + s) * embed_dim + d];
                }
            }
        }
    }
    optimizer_->update(token_embedding_.weight(), d_emb);
    
    TrainStep step;
    step.loss = avg_loss;
    step.perplexity = perplexity;
    step.learning_rate = config_.learning_rate;
    step.gradient_norm = compute_gradient_norm();
    
    optimizer_step_count_++;
    return step;
}

std::vector<Tensor*> LLMModel::get_parameters() {
    std::vector<Tensor*> params;
    
    // Embedding layer
    params.push_back(&token_embedding_.weight());
    
    // Transformer layers
    for (auto& layer : transformer_layers_) {
        params.push_back(&layer.norm1().weight());
        params.push_back(&layer.norm1().bias());
        params.push_back(&layer.attn().q_linear().weight());
        params.push_back(&layer.attn().q_linear().bias());
        params.push_back(&layer.attn().k_linear().weight());
        params.push_back(&layer.attn().k_linear().bias());
        params.push_back(&layer.attn().v_linear().weight());
        params.push_back(&layer.attn().v_linear().bias());
        params.push_back(&layer.attn().out_linear().weight());
        params.push_back(&layer.attn().out_linear().bias());
        params.push_back(&layer.norm2().weight());
        params.push_back(&layer.norm2().bias());
        params.push_back(&layer.ffn().linear1().weight());
        params.push_back(&layer.ffn().linear1().bias());
        params.push_back(&layer.ffn().linear2().weight());
        params.push_back(&layer.ffn().linear2().bias());
    }
    
    // Final norm
    params.push_back(&final_norm_.weight());
    params.push_back(&final_norm_.bias());
    
    // LM head
    params.push_back(&lm_head_.weight());
    params.push_back(&lm_head_.bias());
    
    return params;
}

void LLMModel::apply_gradient_update(float learning_rate) {
    // Parameter updates are handled analytically with exact backpropagation in training_step()
}

LLMModel::TrainingStats LLMModel::train_epoch(const std::vector<Tensor>& input_batches,
                                              const std::vector<Tensor>& target_batches,
                                              size_t epoch,
                                              size_t total_epochs) {
    assert(input_batches.size() == target_batches.size());
    
    TrainingStats stats;
    float total_loss = 0.0f;
    
    size_t num_batches = input_batches.size();
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Update learning rate with scheduler
        float scheduled_lr = lr_scheduler_->get_lr(batch_idx, num_batches);
        config_.learning_rate = scheduled_lr;
        
        // Training step
        auto step = training_step(input_batches[batch_idx], target_batches[batch_idx]);
        
        total_loss += step.loss;
        stats.loss = step.loss;
        stats.learning_rate = step.learning_rate;
        stats.gradient_norm = step.gradient_norm;
        stats.step = epoch * num_batches + batch_idx;
        
        // Print progress every 10 batches
        if ((batch_idx + 1) % 10 == 0 || batch_idx == 0) {
            float avg_loss = total_loss / (batch_idx + 1);
            std::cout << "Epoch " << epoch + 1 << "/" << total_epochs 
                      << " | Batch " << batch_idx + 1 << "/" << num_batches
                      << " | Loss: " << step.loss 
                      << " | Avg Loss: " << avg_loss
                      << " | LR: " << std::scientific << scheduled_lr << std::defaultfloat
                      << std::endl;
        }
    }
    
    stats.avg_loss = total_loss / num_batches;
    stats.perplexity = std::exp(stats.avg_loss);
    
    return stats;
}

void LLMModel::update_learning_rate(size_t current_step, size_t total_steps) {
    float new_lr = lr_scheduler_->get_lr(current_step, total_steps);
    config_.learning_rate = new_lr;
    
    // Update optimizer's learning rate
    optimizer_->set_learning_rate(new_lr);
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
        
        // Apply temperature and a light repetition penalty so generation does not
        // collapse immediately to the same token under weak prototype weights.
        const float temperature = std::max(config_.temperature, 1e-5f);
        std::vector<float> adjusted_logits(config_.vocab_size);
        for (size_t v = 0; v < config_.vocab_size; ++v) {
            adjusted_logits[v] = last_logits[v] / temperature;
        }

        const size_t repetition_window = std::min<size_t>(sequence.size(), 32);
        for (size_t r = 0; r < repetition_window; ++r) {
            const int token_id = sequence[sequence.size() - 1 - r];
            if (token_id < 0 || token_id >= static_cast<int>(config_.vocab_size)) {
                continue;
            }

            float& logit = adjusted_logits[static_cast<size_t>(token_id)];
            if (logit > 0.0f) {
                logit /= 1.2f;
            } else {
                logit *= 1.2f;
            }
        }

        // Sample from top-k candidates instead of greedy argmax.
        const size_t top_k = std::min(
            config_.top_k == 0 ? config_.vocab_size : config_.top_k,
            config_.vocab_size);

        std::vector<size_t> candidates(config_.vocab_size);
        std::iota(candidates.begin(), candidates.end(), size_t{0});
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
            candidates.end(),
            [&](size_t lhs, size_t rhs) {
                return adjusted_logits[lhs] > adjusted_logits[rhs];
            });

        float max_logit = adjusted_logits[candidates[0]];
        for (size_t i = 1; i < top_k; ++i) {
            max_logit = std::max(max_logit, adjusted_logits[candidates[i]]);
        }

        std::vector<double> weights(top_k, 0.0);
        for (size_t i = 0; i < top_k; ++i) {
            weights[i] = static_cast<double>(std::exp(adjusted_logits[candidates[i]] - max_logit));
        }

        int next_token = static_cast<int>(candidates[0]);
        const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (total_weight > 0.0) {
            std::discrete_distribution<size_t> sample_dist(weights.begin(), weights.end());
            next_token = static_cast<int>(candidates[sample_dist(gen)]);
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
    // Compute the norm of gradients (approximation based on parameter values)
    float total_norm = 0.0f;
    
    // Embedding
    const float* emb_data = token_embedding_.weight().data();
    size_t sample_size = std::min(size_t(100), token_embedding_.weight().elem_count());
    for (size_t i = 0; i < sample_size; ++i) {
        total_norm += emb_data[i] * emb_data[i];
    }
    
    // LM head
    const float* head_data = lm_head_.weight().data();
    sample_size = std::min(size_t(100), lm_head_.weight().elem_count());
    for (size_t i = 0; i < sample_size; ++i) {
        total_norm += head_data[i] * head_data[i];
    }
    
    return std::sqrt(total_norm + 1e-8f);  // Add epsilon to avoid sqrt(0)
}

void LLMModel::clip_gradients(float max_norm) {
    // Gradient clipping: if ||g|| > max_norm, scale g by max_norm / ||g||
    float grad_norm = compute_gradient_norm();
    
    if (grad_norm > max_norm && grad_norm > 1e-5f) {
        float scale = max_norm / grad_norm;
        
        // Apply scaling to key parameters
        float* emb_data = token_embedding_.weight().data();
        for (size_t i = 0; i < token_embedding_.weight().elem_count(); ++i) {
            emb_data[i] *= scale;
        }
    }
}

void LLMModel::zero_gradients() {
    // This is a simplified version - in real backprop, we'd clear gradient accumulators
    // Here, we just note that gradients are "zeroed" conceptually
}

void LLMModel::optimizer_step(float learning_rate) {
    // Perform an optimizer step with the given learning rate
    apply_gradient_update(learning_rate);
    optimizer_step_count_++;
}

}  // namespace nks_llm
