#ifndef NKS_LLM_TENSOR_H
#define NKS_LLM_TENSOR_H

#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

namespace nks_llm {

/**
 * ============================================================
 *  Tensor Class — Efficient N-dimensional array for LLM
 *  
 *  Memory layout: Row-major (C-style)
 *  No external dependencies, pure C++17
 * ============================================================
 */
class Tensor {
public:
    using value_type = float;

    // Constructors
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape, bool zero_init = true);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Shape info
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t elem_count() const { return size_; }

    // Access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    float& operator[](size_t idx) { return data_[idx]; }
    const float& operator[](size_t idx) const { return data_[idx]; }

    // Reshape (no copy)
    void reshape(const std::vector<size_t>& new_shape);

    // Operators (element-wise)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Scalar ops
    Tensor operator+(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor& operator+=(float scalar);
    Tensor& operator*=(float scalar);

    // Matrix operations
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor transpose(const Tensor& t, size_t axis0, size_t axis1);

    // Activation functions
    static Tensor relu(const Tensor& t);
    static Tensor gelu(const Tensor& t);
    static Tensor softmax(const Tensor& t, int axis = -1);
    static Tensor sigmoid(const Tensor& t);

    // Reductions
    Tensor sum(int axis = -1, bool keep_dims = false) const;
    Tensor mean(int axis = -1, bool keep_dims = false) const;
    Tensor max(int axis = -1, bool keep_dims = false) const;

    // Normalization
    static Tensor layer_norm(const Tensor& t, float eps = 1e-5);
    static void batch_norm(Tensor& t, float momentum = 0.1f, float eps = 1e-5);

    // Initialization
    void uniform_(float a, float b);
    void normal_(float mean, float std);
    void xavier_uniform_();
    void kaiming_uniform_();
    void zeros_();
    void ones_();

    // I/O
    bool save_binary(const std::string& path) const;
    bool load_binary(const std::string& path);

    // Utility
    void print(const std::string& name = "", size_t max_elems = 12) const;
    float norm() const;
    float sum() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    size_t size_ = 0;

    void allocate();
    std::vector<size_t> broadcast_shape(const Tensor& other) const;
    int normalize_axis(int axis) const;
    static float gelu_approx(float x);
};

}  // namespace nks_llm

#endif  // NKS_LLM_TENSOR_H
