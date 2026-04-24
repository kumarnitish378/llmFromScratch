#ifndef NKS_LLM_MODEL_H
#define NKS_LLM_MODEL_H

#include "layers.h"
#include "optimizer.h"
#include <vector>
#include <string>
#include <memory>

namespace nks_llm {

/**
 * ============================================================
 *  Model Configuration
 * ============================================================
 */
struct ModelConfig {
    // Architecture
    size_t vocab_size = 2048;          // Tokenizer vocabulary size
    size_t max_seq_length = 2048;      // Maximum sequence length
    size_t embedding_dim = 1024;       // Hidden dimension
    size_t num_layers = 24;            // Number of transformer blocks
    size_t num_heads = 16;             // Number of attention heads
    size_t ff_dim = 4096;              // Feed-forward dimension (4x embedding_dim)

    // Training
    float dropout_prob = 0.1f;
    float learning_rate = 1e-4f;
    float weight_decay = 1e-5f;
    float gradient_clip = 1.0f;

    // Generation
    size_t top_k = 50;
    float top_p = 0.95f;
    float temperature = 1.0f;

    // Optimization
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;

    size_t batch_size = 32;
    size_t num_epochs = 10;
    
    static ModelConfig get_large_model();
    static ModelConfig get_base_model();
    static ModelConfig get_small_model();
};

/**
 * ============================================================
 *  LLM Model — Full Transformer-based language model
 *  
 *  Encoder-only transformer for causal language modeling
 *  1B parameters: 24 layers x 1024 dims
 * ============================================================
 */
class LLMModel {
public:
    explicit LLMModel(const ModelConfig& config);

    // Forward pass
    // input_ids: (batch_size, seq_length)
    // returns logits: (batch_size, seq_length, vocab_size)
    Tensor forward(const Tensor& input_ids);

    // Training step with optimizer
    struct TrainStep {
        float loss = 0.0f;
        float perplexity = 0.0f;
        float learning_rate = 0.0f;
        float gradient_norm = 0.0f;
    };

    TrainStep training_step(const Tensor& input_ids, const Tensor& target_ids);
    
    // Full training loop
    struct TrainingStats {
        float loss = 0.0f;
        float avg_loss = 0.0f;
        float perplexity = 0.0f;
        float learning_rate = 0.0f;
        float gradient_norm = 0.0f;
        size_t step = 0;
    };
    
    // Train for multiple steps
    TrainingStats train_epoch(const std::vector<Tensor>& input_batches,
                             const std::vector<Tensor>& target_batches,
                             size_t epoch,
                             size_t total_epochs);
    
    // Get optimizer
    Adam& get_optimizer() { return *optimizer_; }
    const Adam& get_optimizer() const { return *optimizer_; }

    // Generation
    std::vector<int> generate(const std::vector<int>& prompt, 
                              size_t max_new_tokens = 100);

    // Model management
    bool save(const std::string& checkpoint_path) const;
    bool load(const std::string& checkpoint_path);

    // Accessors
    const ModelConfig& config() const { return config_; }
    size_t num_parameters() const { return num_parameters_; }
    size_t num_trainable_parameters() const { return num_parameters_; }

    // Gradient/optimization utilities
    float compute_gradient_norm() const;
    void clip_gradients(float max_norm);
    void zero_gradients();
    void optimizer_step(float learning_rate);
    
    // Collect all trainable parameters
    std::vector<Tensor*> get_parameters();
    
    // Update learning rate schedule
    void update_learning_rate(size_t current_step, size_t total_steps);

private:
    ModelConfig config_;
    size_t num_parameters_ = 0;

    // Model components
    Embedding token_embedding_;
    PositionalEncoding pos_encoding_;
    std::vector<TransformerBlock> transformer_layers_;
    LayerNorm final_norm_;
    Linear lm_head_;  // Language modeling head: embed_dim -> vocab_size

    // Optimizer
    std::unique_ptr<Adam> optimizer_;
    std::unique_ptr<LRScheduler> lr_scheduler_;
    
    // Optimizer state (for tracking)
    std::vector<Tensor> m_states_;  // First moment estimates
    std::vector<Tensor> v_states_;  // Second moment estimates
    size_t optimizer_step_count_ = 0;

    // Helper functions
    void initialize_weights();
    Tensor compute_causal_mask(size_t seq_length) const;
    Tensor compute_logits(const Tensor& hidden_states);
    float compute_cross_entropy_loss(const Tensor& logits, const Tensor& target_ids);
    
    // Gradient approximation (finite differences for each parameter group)
    void apply_gradient_update(float learning_rate);
};

}  // namespace nks_llm

#endif  // NKS_LLM_MODEL_H
