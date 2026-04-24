#ifndef NKS_LLM_OPTIMIZER_H
#define NKS_LLM_OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <string>
#include <cmath>

namespace nks_llm {

/**
 * ============================================================
 *  Adam Optimizer — Adaptive Moment Estimation
 *  
 *  Combines the advantages of AdaGrad and RMSProp
 *  Learning rate adapts per parameter based on gradient history
 * ============================================================
 */
class Adam {
public:
    struct Config {
        float learning_rate = 0.001f;
        float beta1 = 0.9f;           // Exponential decay for 1st moment
        float beta2 = 0.999f;         // Exponential decay for 2nd moment
        float epsilon = 1e-8f;        // Numerical stability
        float weight_decay = 0.0f;    // L2 regularization
        float gradient_clip = 1.0f;   // Gradient clipping threshold
    };

    Adam();
    explicit Adam(const Config& config);

    // Update parameters: param -= alpha * param_grad / (sqrt(v) + eps)
    void update(Tensor& param, const Tensor& param_grad);
    
    // Update multiple parameters (typical usage)
    void step(std::vector<Tensor>& params, 
              const std::vector<Tensor>& grads);

    // Reset optimizer state
    void reset();

    // Set learning rate schedule
    void set_learning_rate(float lr) { config_.learning_rate = lr; }
    
    // Get current step count
    size_t get_step() const { return step_; }

    // Learning rate scheduling
    float get_scheduled_lr(size_t total_steps) const;

private:
    Config config_;
    size_t step_ = 0;
    
    std::vector<Tensor> m_;  // First moment (mean)
    std::vector<Tensor> v_;  // Second moment (variance)
    
    bool initialized_ = false;

    // Helper: Initialize moment states if needed
    void ensure_initialized(size_t num_params);
};

/**
 * ============================================================
 *  Learning Rate Scheduler
 * ============================================================
 */
class LRScheduler {
public:
    enum ScheduleType {
        CONSTANT,
        LINEAR_WARMUP,
        COSINE_ANNEALING,
        EXPONENTIAL_DECAY,
    };

    LRScheduler(float initial_lr, ScheduleType schedule_type = CONSTANT);

    float get_lr(size_t current_step, size_t total_steps);

    void set_warmup_steps(size_t steps) { warmup_steps_ = steps; }
    
private:
    float initial_lr_;
    ScheduleType schedule_type_;
    size_t warmup_steps_ = 1000;

    float linear_warmup_decay(size_t current_step, size_t total_steps);
    float cosine_annealing(size_t current_step, size_t total_steps);
    float exponential_decay(size_t current_step);
};

}  // namespace nks_llm

#endif  // NKS_LLM_OPTIMIZER_H
