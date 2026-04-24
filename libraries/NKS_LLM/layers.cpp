#include "layers.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace nks_llm {

// ============================================================
//  Linear Layer Implementation
// ============================================================

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      weight_({out_features, in_features}),
      bias_({out_features}),
      has_bias_(bias) {
    // Xavier initialization for weights
    weight_.xavier_uniform_();
    if (has_bias_) {
        bias_.zeros_();
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input shape: (..., in_features)
    // weight shape: (out_features, in_features)
    // output: (..., out_features)
    
    Tensor output = Tensor::matmul(input, Tensor::transpose(weight_, 0, 1));
    
    if (has_bias_) {
        output += bias_;
    }
    
    return output;
}

Tensor Linear::forward(const Tensor& input) const {
    Tensor output = Tensor::matmul(input, Tensor::transpose(weight_, 0, 1));
    if (has_bias_) {
        output += bias_;
    }
    return output;
}

bool Linear::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&in_features_), sizeof(in_features_));
    file.write(reinterpret_cast<const char*>(&out_features_), sizeof(out_features_));
    
    weight_.save_binary(path + ".weight");
    if (has_bias_) {
        bias_.save_binary(path + ".bias");
    }
    
    return file.good();
}

bool Linear::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&in_features_), sizeof(in_features_));
    file.read(reinterpret_cast<char*>(&out_features_), sizeof(out_features_));
    
    weight_.load_binary(path + ".weight");
    if (has_bias_) {
        bias_.load_binary(path + ".bias");
    }
    
    return file.good();
}

// ============================================================
//  Embedding Layer Implementation
// ============================================================

Embedding::Embedding(size_t vocab_size, size_t embedding_dim)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      weight_({vocab_size, embedding_dim}) {
    // Normal initialization with std = sqrt(1 / embedding_dim)
    float std = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    weight_.normal_(0.0f, std);
}

Tensor Embedding::forward(const Tensor& input_ids) {
    // input_ids shape: (...,) with values in [0, vocab_size)
    // output shape: (..., embedding_dim)
    
    assert(input_ids.ndim() >= 1);
    
    std::vector<size_t> out_shape = input_ids.shape();
    out_shape.push_back(embedding_dim_);
    
    Tensor output(out_shape);
    
    // Lookup embeddings
    const float* weight_ptr = weight_.data();
    float* out_ptr = output.data();
    
    for (size_t i = 0; i < input_ids.elem_count(); ++i) {
        int token_id = static_cast<int>(input_ids[i]);
        assert(token_id >= 0 && token_id < static_cast<int>(vocab_size_));
        
        // Copy embedding for this token
        const float* embedding = weight_ptr + token_id * embedding_dim_;
        std::copy(embedding, embedding + embedding_dim_, 
                  out_ptr + i * embedding_dim_);
    }
    
    return output;
}

Tensor Embedding::forward(const Tensor& input_ids) const {
    std::vector<size_t> out_shape = input_ids.shape();
    out_shape.push_back(embedding_dim_);
    
    Tensor output(out_shape);
    
    const float* weight_ptr = weight_.data();
    float* out_ptr = output.data();
    
    for (size_t i = 0; i < input_ids.elem_count(); ++i) {
        int token_id = static_cast<int>(input_ids[i]);
        assert(token_id >= 0 && token_id < static_cast<int>(vocab_size_));
        
        const float* embedding = weight_ptr + token_id * embedding_dim_;
        std::copy(embedding, embedding + embedding_dim_,
                  out_ptr + i * embedding_dim_);
    }
    
    return output;
}

bool Embedding::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    file.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(embedding_dim_));
    
    weight_.save_binary(path + ".weight");
    return file.good();
}

bool Embedding::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&vocab_size_), sizeof(vocab_size_));
    file.read(reinterpret_cast<char*>(&embedding_dim_), sizeof(embedding_dim_));
    
    weight_.load_binary(path + ".weight");
    return file.good();
}

// ============================================================
//  Positional Encoding Implementation
// ============================================================

PositionalEncoding::PositionalEncoding(size_t embedding_dim, size_t max_seq_length)
    : embedding_dim_(embedding_dim),
      encoding_({max_seq_length, embedding_dim}) {
    
    // Compute sinusoidal positional encodings
    float* enc_ptr = encoding_.data();
    
    for (size_t pos = 0; pos < max_seq_length; ++pos) {
        for (size_t i = 0; i < embedding_dim; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / embedding_dim);
            
            if (i % 2 == 0) {
                // Even indices: sin
                enc_ptr[pos * embedding_dim + i] = std::sin(angle);
            } else {
                // Odd indices: cos
                enc_ptr[pos * embedding_dim + i] = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::forward(size_t seq_length) {
    assert(seq_length <= encoding_.shape()[0]);
    
    // Return first seq_length positional encodings
    std::vector<size_t> shape = {seq_length, embedding_dim_};
    Tensor result(shape);
    
    std::copy(encoding_.data(), encoding_.data() + seq_length * embedding_dim_,
              result.data());
    
    return result;
}

// ============================================================
//  LayerNorm Implementation
// ============================================================

LayerNorm::LayerNorm(size_t normalized_shape, float eps)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      weight_({normalized_shape}),
      bias_({normalized_shape}) {
    weight_.ones_();
    bias_.zeros_();
}

Tensor LayerNorm::forward(const Tensor& input) {
    return Tensor::layer_norm(input, eps_) * weight_ + bias_;
}

Tensor LayerNorm::forward(const Tensor& input) const {
    return Tensor::layer_norm(input, eps_) * weight_ + bias_;
}

bool LayerNorm::save(const std::string& path) const {
    weight_.save_binary(path + ".weight");
    bias_.save_binary(path + ".bias");
    return true;
}

bool LayerNorm::load(const std::string& path) {
    weight_.load_binary(path + ".weight");
    bias_.load_binary(path + ".bias");
    return true;
}

// ============================================================
//  MultiHeadAttention Implementation
// ============================================================

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads, float dropout_p)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      dropout_p_(dropout_p),
      q_linear_(embed_dim, embed_dim),
      k_linear_(embed_dim, embed_dim),
      v_linear_(embed_dim, embed_dim),
      out_linear_(embed_dim, embed_dim) {
    
    assert(embed_dim % num_heads == 0);
    head_dim_ = embed_dim / num_heads;
}

Tensor MultiHeadAttention::scaled_dot_product_attention(
    const Tensor& q,  // (batch*num_heads, seq_len_q, head_dim)
    const Tensor& k,  // (batch*num_heads, seq_len_k, head_dim)
    const Tensor& v,  // (batch*num_heads, seq_len_v, head_dim)
    const Tensor* mask) {
    
    assert(q.shape().size() == 3);
    
    // Compute attention scores: Q @ K^T / sqrt(d_k)
    Tensor k_t = Tensor::transpose(k, 1, 2);  // (batch*num_heads, head_dim, seq_len_k)
    Tensor scores = Tensor::matmul(q, k_t);   // (batch*num_heads, seq_len_q, seq_len_k)
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    scores *= scale;
    
    // Apply attention mask if provided
    if (mask != nullptr) {
        for (size_t i = 0; i < scores.elem_count(); ++i) {
            if ((*mask)[i] == 0.0f) {
                scores[i] = -1e9f;  // Large negative for masked positions
            }
        }
    }
    
    // Apply softmax
    Tensor attn_weights = Tensor::softmax(scores, -1);
    
    // Apply attention weights to values
    Tensor output = Tensor::matmul(attn_weights, v);  // (batch*num_heads, seq_len_q, head_dim)
    
    attention_weights_ = attn_weights;
    return output;
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                   const Tensor* attention_mask) {
    // For simplicity in initial implementation, apply attention to entire tensor
    // query, key, value shape: (batch, seq_len, embed_dim)
    assert(query.ndim() == 3);
    
    // Project Q, K, V
    Tensor q = q_linear_.forward(query);  // (batch, seq_len_q, embed_dim)
    Tensor k = k_linear_.forward(key);    // (batch, seq_len_k, embed_dim)
    Tensor v = v_linear_.forward(value);  // (batch, seq_len_k, embed_dim)
    
    // Simplified: for now, just apply output projection without reshaping
    // In a full implementation, we'd reshape for multi-head attention
    
    // For this prototype, treat as single "super-head"
    // Compute Q @ K^T attention
    size_t batch_size = q.shape()[0];
    size_t seq_len_q = q.shape()[1];
    size_t seq_len_k = k.shape()[1];
    
    Tensor k_t = Tensor::transpose(k, 1, 2);  // (batch, embed_dim, seq_len_k)
    Tensor scores = Tensor::matmul(q, k_t);   // (batch, seq_len_q, seq_len_k)
    
    float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));
    scores *= scale;
    
    // Apply softmax
    Tensor attn_weights = Tensor::softmax(scores, -1);
    
    // Apply to values
    Tensor attn_output = Tensor::matmul(attn_weights, v);  // (batch, seq_len_q, embed_dim)
    
    // Output projection
    Tensor output = out_linear_.forward(attn_output);
    
    return output;
}

bool MultiHeadAttention::save(const std::string& path) const {
    q_linear_.save(path + ".q");
    k_linear_.save(path + ".k");
    v_linear_.save(path + ".v");
    out_linear_.save(path + ".out");
    return true;
}

bool MultiHeadAttention::load(const std::string& path) {
    q_linear_.load(path + ".q");
    k_linear_.load(path + ".k");
    v_linear_.load(path + ".v");
    out_linear_.load(path + ".out");
    return true;
}

// ============================================================
//  FeedForward Implementation
// ============================================================

FeedForward::FeedForward(size_t d_model, size_t d_ff)
    : linear1_(d_model, d_ff),
      linear2_(d_ff, d_model) {}

Tensor FeedForward::forward(const Tensor& input) {
    Tensor hidden = Tensor::gelu(linear1_.forward(input));
    return linear2_.forward(hidden);
}

Tensor FeedForward::forward(const Tensor& input) const {
    Tensor hidden = Tensor::gelu(linear1_.forward(input));
    return linear2_.forward(hidden);
}

bool FeedForward::save(const std::string& path) const {
    linear1_.save(path + ".linear1");
    linear2_.save(path + ".linear2");
    return true;
}

bool FeedForward::load(const std::string& path) {
    linear1_.load(path + ".linear1");
    linear2_.load(path + ".linear2");
    return true;
}

// ============================================================
//  TransformerBlock Implementation
// ============================================================

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim,
                                   float dropout_p)
    : norm1_(embed_dim),
      attn_(embed_dim, num_heads, dropout_p),
      norm2_(embed_dim),
      ffn_(embed_dim, ff_dim),
      dropout_p_(dropout_p) {}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor* attention_mask) {
    // Pre-norm architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    
    // Self-attention with residual connection
    Tensor x_norm = norm1_.forward(x);
    Tensor attn_out = attn_.forward(x_norm, x_norm, x_norm, attention_mask);
    Tensor x_after_attn = x + attn_out;  // Residual
    
    // Feed-forward with residual connection
    Tensor x_norm2 = norm2_.forward(x_after_attn);
    Tensor ffn_out = ffn_.forward(x_norm2);
    Tensor output = x_after_attn + ffn_out;  // Residual
    
    return output;
}

bool TransformerBlock::save(const std::string& path) const {
    norm1_.save(path + ".norm1");
    attn_.save(path + ".attn");
    norm2_.save(path + ".norm2");
    ffn_.save(path + ".ffn");
    return true;
}

bool TransformerBlock::load(const std::string& path) {
    norm1_.load(path + ".norm1");
    attn_.load(path + ".attn");
    norm2_.load(path + ".norm2");
    ffn_.load(path + ".ffn");
    return true;
}

}  // namespace nks_llm
