#include "gpu_backend.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace nks_llm {
namespace gpu_backend {
namespace {

constexpr int kBlockSize = 16;
constexpr int kSoftmaxBlockSize = 256;

bool ok(cudaError_t status) {
    return status == cudaSuccess;
}

__global__ void matmul_kernel(const float* a, const float* b, float* out,
                              std::size_t total_batches,
                              std::size_t m, std::size_t n, std::size_t p,
                              bool b_is_batched) {
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t batch = blockIdx.z;

    if (batch >= total_batches || row >= m || col >= p) {
        return;
    }

    const float* a_batch = a + batch * m * n;
    const float* b_batch = b + (b_is_batched ? batch * n * p : 0);
    float sum = 0.0f;

    for (std::size_t k = 0; k < n; ++k) {
        sum += a_batch[row * n + k] * b_batch[k * p + col];
    }

    out[batch * m * p + row * p + col] = sum;
}

__global__ void softmax_kernel(float* data, std::size_t outer_size, std::size_t inner_size) {
    const std::size_t row = blockIdx.x;
    if (row >= outer_size) {
        return;
    }

    extern __shared__ float shared[];
    float* max_values = shared;
    float* sums = shared;
    float* row_data = data + row * inner_size;

    float local_max = -3.402823466e+38F;
    for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }

    max_values[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            max_values[threadIdx.x] = fmaxf(max_values[threadIdx.x], max_values[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float row_max = max_values[0];
    float local_sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        const float value = expf(row_data[i] - row_max);
        row_data[i] = value;
        local_sum += value;
    }

    sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float row_sum = sums[0];
    if (row_sum == 0.0f) {
        return;
    }

    for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}

bool copy_matmul_run_copy_back(const float* a, const float* b, float* out,
                               std::size_t batch, std::size_t m, std::size_t n, std::size_t p,
                               bool b_is_batched) {
    const std::size_t a_bytes = batch * m * n * sizeof(float);
    const std::size_t b_batches = b_is_batched ? batch : 1;
    const std::size_t b_bytes = b_batches * n * p * sizeof(float);
    const std::size_t out_bytes = batch * m * p * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;

    bool success =
        ok(cudaMalloc(&d_a, a_bytes)) &&
        ok(cudaMalloc(&d_b, b_bytes)) &&
        ok(cudaMalloc(&d_out, out_bytes)) &&
        ok(cudaMemcpy(d_a, a, a_bytes, cudaMemcpyHostToDevice)) &&
        ok(cudaMemcpy(d_b, b, b_bytes, cudaMemcpyHostToDevice));

    if (success) {
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y, batch);
        matmul_kernel<<<grid, block>>>(d_a, d_b, d_out, batch, m, n, p, b_is_batched);
        success = ok(cudaGetLastError()) &&
                  ok(cudaDeviceSynchronize()) &&
                  ok(cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return success;
}

}  // namespace

bool is_available() {
    int count = 0;
    return ok(cudaGetDeviceCount(&count)) && count > 0;
}

const char* backend_name() {
    return is_available() ? "CUDA" : "CPU";
}

bool matmul_2d(const float* a, const float* b, float* out,
               std::size_t m, std::size_t n, std::size_t p) {
    if (!is_available()) {
        return false;
    }
    return copy_matmul_run_copy_back(a, b, out, 1, m, n, p, false);
}

bool matmul_3d_2d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p) {
    if (!is_available()) {
        return false;
    }
    return copy_matmul_run_copy_back(a, b, out, batch, m, n, p, false);
}

bool matmul_3d_3d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p) {
    if (!is_available()) {
        return false;
    }
    return copy_matmul_run_copy_back(a, b, out, batch, m, n, p, true);
}

bool softmax_last_dim(float* data, std::size_t outer_size, std::size_t inner_size) {
    if (!is_available() || outer_size == 0 || inner_size == 0) {
        return false;
    }

    const std::size_t bytes = outer_size * inner_size * sizeof(float);
    float* d_data = nullptr;
    if (!ok(cudaMalloc(&d_data, bytes))) {
        return false;
    }

    bool success = ok(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
    if (success) {
        softmax_kernel<<<static_cast<unsigned int>(outer_size), kSoftmaxBlockSize,
                         kSoftmaxBlockSize * sizeof(float)>>>(d_data, outer_size, inner_size);
        success = ok(cudaGetLastError()) &&
                  ok(cudaDeviceSynchronize()) &&
                  ok(cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(d_data);
    return success;
}

}  // namespace gpu_backend
}  // namespace nks_llm
