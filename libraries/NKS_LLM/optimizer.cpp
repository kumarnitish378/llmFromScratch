#include "optimizer.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nks_llm {

// ============================================================
//  Adam Implementation
// ============================================================

Adam::Adam()
    : config_(Config()) {
}

Adam::Adam(const Config& config)
    : config_(config) {
}

void Adam::ensure_initialized(size_t num_params) {
    if (initialized_ && m_.size() == num_params) {
        return;  // Already initialized with correct size
    }
    
    m_.clear();
    v_.clear();
    
    // We'll lazily initialize moment states per parameter
    initialized_ = true;
}

void Adam::reset() {
    m_.clear();
    v_.clear();
    step_ = 0;
    initialized_ = false;
}

void Adam::update(Tensor& param, const Tensor& param_grad) {
    assert(param.shape() == param_grad.shape());
    
    step_++;
    
    // Ensure m and v are initialized
    if (m_.empty() || m_[0].shape() != param.shape()) {
        m_.clear();
        v_.clear();
        m_.push_back(Tensor(param.shape(), true));
        v_.push_back(Tensor(param.shape(), true));
        m_[0].zeros_();
        v_[0].zeros_();
    }
    
    float* param_ptr = param.data();
    const float* grad_ptr = param_grad.data();
    float* m_ptr = m_[0].data();
    float* v_ptr = v_[0].data();
    
    size_t size = param.elem_count();
    
    const float beta1 = config_.beta1;
    const float beta2 = config_.beta2;
    const float eps = config_.epsilon;
    const float lr = config_.learning_rate;
    const float weight_decay = config_.weight_decay;
    
    // Compute bias-corrected learning rate
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_));
    float corrected_lr = lr * std::sqrt(bias_correction2) / bias_correction1;
    
    for (size_t i = 0; i < size; ++i) {
        float g = grad_ptr[i];
        
        // Gradient clipping
        if (config_.gradient_clip > 0.0f) {
            g = std::max(-config_.gradient_clip, std::min(g, config_.gradient_clip));
        }
        
        // Weight decay (L2 regularization)
        if (weight_decay > 0.0f) {
            g += weight_decay * param_ptr[i];
        }
        
        // Update biased first moment estimate
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
        
        // Update biased second raw moment estimate
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
        
        // Update parameter
        param_ptr[i] -= corrected_lr * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps);
    }
}

void Adam::step(std::vector<Tensor>& params, 
                const std::vector<Tensor>& grads) {
    assert(params.size() == grads.size());
    
    for (size_t i = 0; i < params.size(); ++i) {
        update(params[i], grads[i]);
    }
}

// ============================================================
//  LRScheduler Implementation
// ============================================================

LRScheduler::LRScheduler(float initial_lr, ScheduleType schedule_type)
    : initial_lr_(initial_lr), schedule_type_(schedule_type) {
}

float LRScheduler::get_lr(size_t current_step, size_t total_steps) {
    switch (schedule_type_) {
        case LINEAR_WARMUP:
            return linear_warmup_decay(current_step, total_steps);
        case COSINE_ANNEALING:
            return cosine_annealing(current_step, total_steps);
        case EXPONENTIAL_DECAY:
            return exponential_decay(current_step);
        case CONSTANT:
        default:
            return initial_lr_;
    }
}

float LRScheduler::linear_warmup_decay(size_t current_step, size_t total_steps) {
    if (current_step < warmup_steps_) {
        // Linear warmup
        return initial_lr_ * (static_cast<float>(current_step) / static_cast<float>(warmup_steps_));
    } else {
        // Linear decay after warmup
        size_t steps_after_warmup = current_step - warmup_steps_;
        size_t decay_steps = total_steps - warmup_steps_;
        if (decay_steps == 0) return initial_lr_;
        return initial_lr_ * (1.0f - static_cast<float>(steps_after_warmup) / static_cast<float>(decay_steps));
    }
}

float LRScheduler::cosine_annealing(size_t current_step, size_t total_steps) {
    float progress = static_cast<float>(current_step) / static_cast<float>(total_steps);
    // Cosine annealing from 1 to 0
    float cosine = 0.5f * (1.0f + std::cos(M_PI * progress));
    return initial_lr_ * cosine;
}

float LRScheduler::exponential_decay(size_t current_step) {
    // Decay by 0.96 every 1000 steps
    size_t num_decays = current_step / 1000;
    return initial_lr_ * std::pow(0.96f, static_cast<float>(num_decays));
}

}  // namespace nks_llm
