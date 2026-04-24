#ifndef NKS_LLM_LAYERS_H
#define NKS_LLM_LAYERS_H

#include "tensor.h"
#include <string>

namespace nks_llm {

/**
 * ============================================================
 *  Linear Layer — Affine transformation with bias
 * ============================================================
 */
class Linear {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(const Tensor& input);
    Tensor forward(const Tensor& input) const;

    // Accessors
    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

    // Serialization
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    size_t in_features_;
    size_t out_features_;
    Tensor weight_;  // shape: (out_features, in_features)
    Tensor bias_;    // shape: (out_features,)
    bool has_bias_;
};

/**
 * ============================================================
 *  Embedding Layer — Token embedding lookup table
 * ============================================================
 */
class Embedding {
public:
    Embedding(size_t vocab_size, size_t embedding_dim);

    // Forward: input(ids) -> embeddings(embedding_dim)
    // Input shape: (...,) -> output shape: (..., embedding_dim)
    Tensor forward(const Tensor& input_ids);
    Tensor forward(const Tensor& input_ids) const;

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }

    size_t vocab_size() const { return vocab_size_; }
    size_t embedding_dim() const { return embedding_dim_; }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    Tensor weight_;  // shape: (vocab_size, embedding_dim)
};

/**
 * ============================================================
 *  Positional Encoding — Sinusoidal position embeddings
 * ============================================================
 */
class PositionalEncoding {
public:
    PositionalEncoding(size_t embedding_dim, size_t max_seq_length = 2048);

    // Get positional encoding for sequence of given length
    Tensor forward(size_t seq_length);

    size_t embedding_dim() const { return embedding_dim_; }

private:
    size_t embedding_dim_;
    Tensor encoding_;  // pre-computed encodings
};

/**
 * ============================================================
 *  LayerNorm — Layer normalization with learnable scale/shift
 * ============================================================
 */
class LayerNorm {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5f);

    Tensor forward(const Tensor& input);
    Tensor forward(const Tensor& input) const;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    size_t normalized_shape_;
    float eps_;
    Tensor weight_;  // gamma - learned scale
    Tensor bias_;    // beta - learned shift
};

/**
 * ============================================================
 *  MultiHeadAttention — Self-attention mechanism
 *  
 *  Q, K, V projections -> split into heads -> attention -> concat -> output projection
 * ============================================================
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads, float dropout_p = 0.0f);

    // Forward pass: (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* attention_mask = nullptr);

    Tensor q_proj() { return q_linear_.weight(); }
    Tensor k_proj() { return k_linear_.weight(); }
    Tensor v_proj() { return v_linear_.weight(); }
    Tensor out_proj() { return out_linear_.weight(); }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float dropout_p_;

    Linear q_linear_;
    Linear k_linear_;
    Linear v_linear_;
    Linear out_linear_;

    Tensor attention_weights_;  // Cache for visualization

    Tensor scaled_dot_product_attention(
        const Tensor& q,  // (batch*num_heads, seq_len, head_dim)
        const Tensor& k,
        const Tensor& v,
        const Tensor* mask = nullptr);
};

/**
 * ============================================================
 *  FeedForward Network — Two linear layers with activation
 *  
 *  Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
 * ============================================================
 */
class FeedForward {
public:
    FeedForward(size_t d_model, size_t d_ff);

    Tensor forward(const Tensor& input);
    Tensor forward(const Tensor& input) const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Linear linear1_;  // d_model -> d_ff
    Linear linear2_;  // d_ff -> d_model
};

/**
 * ============================================================
 *  TransformerBlock — Single encoder block
 *  
 *  [LayerNorm] -> Attention -> [Residual] -> LayerNorm -> FFN -> [Residual]
 * ============================================================
 */
class TransformerBlock {
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim,
                     float dropout_p = 0.0f);

    Tensor forward(const Tensor& x, const Tensor* attention_mask = nullptr);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    LayerNorm norm1_;
    MultiHeadAttention attn_;
    LayerNorm norm2_;
    FeedForward ffn_;
    float dropout_p_;
};

}  // namespace nks_llm

#endif  // NKS_LLM_LAYERS_H
