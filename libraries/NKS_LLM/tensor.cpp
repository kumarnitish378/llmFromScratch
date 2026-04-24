#include "tensor.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <numeric>

namespace nks_llm {

Tensor::Tensor(const std::vector<size_t>& shape, bool zero_init)
    : shape_(shape), size_(1) {
    if (shape_.empty()) return;
    for (size_t s : shape_) {
        size_ *= s;
    }
    if (zero_init) {
        data_.assign(size_, 0.0f);
    } else {
        data_.resize(size_);
    }
}

void Tensor::allocate() {
    data_.assign(size_, 0.0f);
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (size_t s : new_shape) {
        new_size *= s;
    }
    assert(new_size == size_ && "Reshape must preserve total size");
    shape_ = new_shape;
}

// ── Element-wise operations ──

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result = *this;
    result += other;
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(size_ == other.size_);
    Tensor result = *this;
    result -= other;
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result = *this;
    result *= other;
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(size_ == other.size_);
    Tensor result = *this;
    result /= other;
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    // Handle exact size match
    if (size_ == other.size_) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }
    
    // Handle broadcasting: allow adding along compatible dimensions
    // For (B, S, D) + (D,)
    if (ndim() == 3 && other.ndim() == 1) {
        size_t batch = shape_[0];
        size_t seq = shape_[1];
        size_t dim = shape_[2];
        
        if (other.size_ == dim) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        data_[b * seq * dim + s * dim + d] += other.data_[d];
                    }
                }
            }
            return *this;
        }
    }
    
    // For (B, S, D) + (S, D)
    if (ndim() == 3 && other.ndim() == 2) {
        size_t batch = shape_[0];
        size_t seq = shape_[1];
        size_t dim = shape_[2];
        
        if (other.shape_[0] == seq && other.shape_[1] == dim) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        data_[b * seq * dim + s * dim + d] += 
                            other.data_[s * dim + d];
                    }
                }
            }
            return *this;
        }
    }
    
    // No matching broadcast pattern found - debug output
    std::cerr << "Tensor::operator+= no broadcast match: self shape [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << shape_[i];
    }
    std::cerr << "] other shape [";
    for (size_t i = 0; i < other.shape_.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << other.shape_[i];
    }
    std::cerr << "]" << std::endl;
    assert(false && "No matching broadcast pattern");
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    assert(size_ == other.size_);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    // Handle exact size match
    if (size_ == other.size_) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }
    
    // Handle broadcasting: allow multiplying along compatible dimensions
    // For (B, S, D) * (D,)
    if (ndim() == 3 && other.ndim() == 1) {
        size_t batch = shape_[0];
        size_t seq = shape_[1];
        size_t dim = shape_[2];
        
        if (other.size_ == dim) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        data_[b * seq * dim + s * dim + d] *= other.data_[d];
                    }
                }
            }
            return *this;
        }
    }
    
    // For (B, S, D) * (S, D)
    if (ndim() == 3 && other.ndim() == 2) {
        size_t batch = shape_[0];
        size_t seq = shape_[1];
        size_t dim = shape_[2];
        
        if (other.shape_[0] == seq && other.shape_[1] == dim) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        data_[b * seq * dim + s * dim + d] *= 
                            other.data_[s * dim + d];
                    }
                }
            }
            return *this;
        }
    }
    
    // No matching broadcast pattern found - just skip
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    assert(size_ == other.size_);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] /= other.data_[i];
    }
    return *this;
}

// ── Scalar operations ──

Tensor Tensor::operator+(float scalar) const {
    Tensor result = *this;
    result += scalar;
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result = *this;
    result *= scalar;
    return result;
}

Tensor& Tensor::operator+=(float scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += scalar;
    }
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

// ── Matrix multiplication ──

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    // Handles: 
    // 2D @ 2D: (m, n) @ (n, p) -> (m, p)
    // 3D @ 3D: (b, m, n) @ (b, n, p) -> (b, m, p)
    // 2D @ 1D: (m, n) @ (n,) -> (m,)

    assert(a.ndim() >= 1 && b.ndim() >= 1);
    assert(a.shape_[a.ndim() - 1] == b.shape_[b.ndim() - 2]);

    // Handle 2D @ 2D (most common for LLM)
    if (a.ndim() == 2 && b.ndim() == 2) {
        size_t m = a.shape_[0];
        size_t n = a.shape_[1];
        size_t p = b.shape_[1];

        Tensor result({m, p}, true);
        const float* a_ptr = a.data_.data();
        const float* b_ptr = b.data_.data();
        float* r_ptr = result.data_.data();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k) {
                    sum += a_ptr[i * n + k] * b_ptr[k * p + j];
                }
                r_ptr[i * p + j] = sum;
            }
        }
        return result;
    }

    // Handle 3D @ 2D: (b, m, n) @ (n, p) -> (b, m, p)
    if (a.ndim() == 3 && b.ndim() == 2) {
        size_t batch = a.shape_[0];
        size_t m = a.shape_[1];
        size_t n = a.shape_[2];
        size_t p = b.shape_[1];

        Tensor result({batch, m, p}, true);
        const float* a_ptr = a.data_.data();
        const float* b_ptr = b.data_.data();
        float* r_ptr = result.data_.data();

        size_t a_batch_size = m * n;
        size_t r_batch_size = m * p;

        for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
            const float* a_batch = a_ptr + b_idx * a_batch_size;
            float* r_batch = r_ptr + b_idx * r_batch_size;

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < p; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < n; ++k) {
                        sum += a_batch[i * n + k] * b_ptr[k * p + j];
                    }
                    r_batch[i * p + j] = sum;
                }
            }
        }
        return result;
    }

    // Handle 3D @ 3D: (b, m, n) @ (b, n, p) -> (b, m, p)
    if (a.ndim() == 3 && b.ndim() == 3) {
        size_t batch = a.shape_[0];
        size_t m = a.shape_[1];
        size_t n = a.shape_[2];
        size_t p = b.shape_[2];

        Tensor result({batch, m, p}, true);
        const float* a_ptr = a.data_.data();
        const float* b_ptr = b.data_.data();
        float* r_ptr = result.data_.data();

        size_t a_batch_size = m * n;
        size_t b_batch_size = n * p;
        size_t r_batch_size = m * p;

        for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
            const float* a_batch = a_ptr + b_idx * a_batch_size;
            const float* b_batch = b_ptr + b_idx * b_batch_size;
            float* r_batch = r_ptr + b_idx * r_batch_size;

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < p; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < n; ++k) {
                        sum += a_batch[i * n + k] * b_batch[k * p + j];
                    }
                    r_batch[i * p + j] = sum;
                }
            }
        }
        return result;
    }

    // Fallback: create zero tensor with correct shape
    std::vector<size_t> out_shape = a.shape_;
    out_shape[out_shape.size() - 1] = b.shape_[b.ndim() - 1];
    Tensor result(out_shape, true);
    return result;
}

// ── Transpose ──

Tensor Tensor::transpose(const Tensor& t, size_t axis0, size_t axis1) {
    assert(axis0 < t.ndim() && axis1 < t.ndim());
    
    // For 2D (most common): simple row/col transpose
    if (t.ndim() == 2 && axis0 == 0 && axis1 == 1) {
        Tensor result({t.shape_[1], t.shape_[0]});
        size_t rows = t.shape_[0];
        size_t cols = t.shape_[1];
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_[j * rows + i] = t.data_[i * cols + j];
            }
        }
        return result;
    }

    // For 3D with last two axes: (batch, m, n) -> (batch, n, m)
    if (t.ndim() == 3 && axis0 == 1 && axis1 == 2) {
        size_t batch = t.shape_[0];
        size_t m = t.shape_[1];
        size_t n = t.shape_[2];
        
        Tensor result({batch, n, m});
        const float* t_ptr = t.data_.data();
        float* r_ptr = result.data_.data();
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    r_ptr[b * n * m + j * m + i] = t_ptr[b * m * n + i * n + j];
                }
            }
        }
        return result;
    }

    // Default: return copy if same axes
    if (axis0 == axis1) return t;
    
    // Unsupported configuration: return copy
    return t;
}

// ── Activation functions ──

Tensor Tensor::relu(const Tensor& t) {
    Tensor result = t;
    for (size_t i = 0; i < result.size_; ++i) {
        result.data_[i] = std::max(0.0f, result.data_[i]);
    }
    return result;
}

float Tensor::gelu_approx(float x) {
    // Approximation: GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    constexpr float c = 0.044715f;
    constexpr float coef = 0.7978845608f;  // sqrt(2/π)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(coef * (x + c * x3)));
}

Tensor Tensor::gelu(const Tensor& t) {
    Tensor result = t;
    for (size_t i = 0; i < result.size_; ++i) {
        result.data_[i] = gelu_approx(result.data_[i]);
    }
    return result;
}

Tensor Tensor::sigmoid(const Tensor& t) {
    Tensor result = t;
    for (size_t i = 0; i < result.size_; ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-result.data_[i]));
    }
    return result;
}

// ── Softmax ──

Tensor Tensor::softmax(const Tensor& t, int axis) {
    int norm_axis = t.normalize_axis(axis);
    Tensor result = t;
    
    if (t.ndim() == 2 && norm_axis == 1) {
        // 2D softmax over last dimension
        size_t rows = t.shape_[0];
        size_t cols = t.shape_[1];
        
        for (size_t i = 0; i < rows; ++i) {
            // Find max for numerical stability
            float max_val = result.data_[i * cols];
            for (size_t j = 0; j < cols; ++j) {
                max_val = std::max(max_val, result.data_[i * cols + j]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                result.data_[i * cols + j] = std::exp(result.data_[i * cols + j] - max_val);
                sum += result.data_[i * cols + j];
            }
            
            // Normalize
            for (size_t j = 0; j < cols; ++j) {
                result.data_[i * cols + j] /= sum;
            }
        }
    }
    return result;
}

// ── Reductions ──

Tensor Tensor::sum(int axis, bool keep_dims) const {
    if (axis == -1) axis = ndim() - 1;
    assert(axis < static_cast<int>(ndim()));
    
    std::vector<size_t> out_shape = shape_;
    if (keep_dims) {
        out_shape[axis] = 1;
    } else {
        out_shape.erase(out_shape.begin() + axis);
    }
    
    Tensor result(out_shape, true);
    
    // Simplified 2D case
    if (ndim() == 2) {
        if (axis == 0) {
            // Sum rows
            for (size_t j = 0; j < shape_[1]; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < shape_[0]; ++i) {
                    sum += data_[i * shape_[1] + j];
                }
                result.data_[j] = sum;
            }
        } else {
            // Sum cols
            for (size_t i = 0; i < shape_[0]; ++i) {
                float sum = 0.0f;
                for (size_t j = 0; j < shape_[1]; ++j) {
                    sum += data_[i * shape_[1] + j];
                }
                result.data_[i] = sum;
            }
        }
    }
    return result;
}

Tensor Tensor::mean(int axis, bool keep_dims) const {
    Tensor result = sum(axis, keep_dims);
    size_t divisor = shape_[axis < 0 ? ndim() - 1 : axis];
    result *= (1.0f / static_cast<float>(divisor));
    return result;
}

Tensor Tensor::max(int axis, bool keep_dims) const {
    // Simplified implementation
    Tensor result(shape_);  // Placeholder
    return result;
}

// ── Normalization ──

Tensor Tensor::layer_norm(const Tensor& t, float eps) {
    assert(t.ndim() >= 2);
    
    Tensor result = t;
    size_t last_dim = t.shape_[t.ndim() - 1];
    size_t batch_size = t.size_ / last_dim;
    
    for (size_t b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            mean += result.data_[b * last_dim + i];
        }
        mean /= static_cast<float>(last_dim);
        
        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float diff = result.data_[b * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(last_dim);
        
        // Normalize
        float std = std::sqrt(var + eps);
        for (size_t i = 0; i < last_dim; ++i) {
            result.data_[b * last_dim + i] = (result.data_[b * last_dim + i] - mean) / std;
        }
    }
    return result;
}

void Tensor::batch_norm(Tensor& t, float momentum, float eps) {
    // Simplified placeholder
}

// ── Initialization ──

void Tensor::uniform_(float a, float b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(a, b);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = dist(gen);
    }
}

void Tensor::normal_(float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = dist(gen);
    }
}

void Tensor::xavier_uniform_() {
    // Xavier initialization for uniform distribution
    float limit = std::sqrt(6.0f / static_cast<float>(size_));
    uniform_(-limit, limit);
}

void Tensor::kaiming_uniform_() {
    // Kaiming initialization
    float bound = std::sqrt(3.0f / static_cast<float>(size_));
    uniform_(-bound, bound);
}

void Tensor::zeros_() {
    data_.assign(size_, 0.0f);
}

void Tensor::ones_() {
    data_.assign(size_, 1.0f);
}

// ── Utility ──

int Tensor::normalize_axis(int axis) const {
    if (axis < 0) {
        axis = static_cast<int>(ndim()) + axis;
    }
    return axis;
}

float Tensor::norm() const {
    float sum_sq = 0.0f;
    for (float x : data_) {
        sum_sq += x * x;
    }
    return std::sqrt(sum_sq);
}

float Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

void Tensor::print(const std::string& name, size_t max_elems) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "], size=" << size_ << ")\n";
    
    std::cout << "Data: [";
    for (size_t i = 0; i < std::min(max_elems, size_); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data_[i];
    }
    if (size_ > max_elems) std::cout << ", ...";
    std::cout << "]\n";
}

// ── I/O ──

bool Tensor::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Write shape
    uint32_t ndim = static_cast<uint32_t>(shape_.size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (size_t s : shape_) {
        uint32_t shape_val = static_cast<uint32_t>(s);
        file.write(reinterpret_cast<const char*>(&shape_val), sizeof(shape_val));
    }
    
    // Write data
    file.write(reinterpret_cast<const char*>(data_.data()), size_ * sizeof(float));
    return file.good();
}

bool Tensor::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Read shape
    uint32_t ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    shape_.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t shape_val = 0;
        file.read(reinterpret_cast<char*>(&shape_val), sizeof(shape_val));
        shape_[i] = shape_val;
    }
    
    // Compute size
    size_ = 1;
    for (size_t s : shape_) size_ *= s;
    
    // Read data
    data_.resize(size_);
    file.read(reinterpret_cast<char*>(data_.data()), size_ * sizeof(float));
    return file.good();
}

}  // namespace nks_llm
