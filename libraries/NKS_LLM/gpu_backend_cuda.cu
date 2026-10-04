#include "gpu_backend.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace nks_llm {
namespace gpu_backend {
namespace {

constexpr int kBlockSize = 16;
constexpr int kSoftmaxBlockSize = 256;

// Tensor storage currently lives on the CPU. Reuse device scratch allocations
// across synchronous calls to avoid cudaMalloc/cudaFree on every matmul.
struct DeviceBuffer {
    float* ptr = nullptr;
    std::size_t capacity_bytes = 0;
};

DeviceBuffer d_a_buffer;
DeviceBuffer d_b_buffer;
DeviceBuffer d_out_buffer;
DeviceBuffer d_softmax_buffer;

bool ok(cudaError_t status) {
    return status == cudaSuccess;
}

bool reserve(DeviceBuffer& buffer, std::size_t bytes) {
    if (bytes <= buffer.capacity_bytes && buffer.ptr != nullptr) {
        return true;
    }

    if (buffer.ptr != nullptr) {
        if (!ok(cudaFree(buffer.ptr))) {
            buffer.ptr = nullptr;
            buffer.capacity_bytes = 0;
            return false;
        }
        buffer.ptr = nullptr;
        buffer.capacity_bytes = 0;
    }

    // Grow geometrically to reduce future reallocations as tensor sizes vary.
    std::size_t capacity = std::max<std::size_t>(bytes, 4096);
    if (buffer.capacity_bytes > 0) {
        capacity = std::max(capacity, buffer.capacity_bytes * 2);
    }
    if (!ok(cudaMalloc(reinterpret_cast<void**>(&buffer.ptr), capacity))) {
        buffer.ptr = nullptr;
        buffer.capacity_bytes = 0;
        return false;
    }
    buffer.capacity_bytes = capacity;
    return true;
}

__global__ void matmul_kernel(const float* a, const float* b, float* out,
                              std::size_t total_batches,
                              std::size_t m, std::size_t n, std::size_t p,
                              bool b_is_batched) {
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t batch = blockIdx.z;

    if (batch >= total_batches || row >= m || col >= p) return;

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
    if (row >= outer_size) return;

    extern __shared__ float shared[];
    float* reductions = shared;
    float* row_data = data + row * inner_size;

    float local_max = -3.402823466e+38F;
    for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    reductions[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float row_max = reductions[0];
    float local_sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        const float value = expf(row_data[i] - row_max);
        row_data[i] = value;
        local_sum += value;
    }
    reductions[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float row_sum = reductions[0];
    if (row_sum > 0.0f) {
        for (std::size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
            row_data[i] /= row_sum;
        }
    }
}

bool copy_matmul_run_copy_back(const float* a, const float* b, float* out,
                               std::size_t batch, std::size_t m, std::size_t n, std::size_t p,
                               bool b_is_batched) {
    if (batch == 0 || m == 0 || n == 0 || p == 0) return false;

    const std::size_t a_bytes = batch * m * n * sizeof(float);
    const std::size_t b_batches = b_is_batched ? batch : 1;
    const std::size_t b_bytes = b_batches * n * p * sizeof(float);
    const std::size_t out_bytes = batch * m * p * sizeof(float);

    if (!reserve(d_a_buffer, a_bytes) ||
        !reserve(d_b_buffer, b_bytes) ||
        !reserve(d_out_buffer, out_bytes)) {
        return false;
    }

    bool success = ok(cudaMemcpy(d_a_buffer.ptr, a, a_bytes, cudaMemcpyHostToDevice)) &&
                   ok(cudaMemcpy(d_b_buffer.ptr, b, b_bytes, cudaMemcpyHostToDevice));
    if (!success) return false;

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y, batch);
    matmul_kernel<<<grid, block>>>(d_a_buffer.ptr, d_b_buffer.ptr, d_out_buffer.ptr,
                                   batch, m, n, p, b_is_batched);
    success = ok(cudaGetLastError()) &&
              ok(cudaDeviceSynchronize()) &&
              ok(cudaMemcpy(out, d_out_buffer.ptr, out_bytes, cudaMemcpyDeviceToHost));
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
    return is_available() && copy_matmul_run_copy_back(a, b, out, 1, m, n, p, false);
}

bool matmul_3d_2d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p) {
    return is_available() && copy_matmul_run_copy_back(a, b, out, batch, m, n, p, false);
}

bool matmul_3d_3d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p) {
    return is_available() && copy_matmul_run_copy_back(a, b, out, batch, m, n, p, true);
}

bool softmax_last_dim(float* data, std::size_t outer_size, std::size_t inner_size) {
    if (!is_available() || outer_size == 0 || inner_size == 0) return false;

    const std::size_t bytes = outer_size * inner_size * sizeof(float);
    if (!reserve(d_softmax_buffer, bytes)) return false;
    if (!ok(cudaMemcpy(d_softmax_buffer.ptr, data, bytes, cudaMemcpyHostToDevice))) return false;

    softmax_kernel<<<static_cast<unsigned int>(outer_size), kSoftmaxBlockSize,
                     kSoftmaxBlockSize * sizeof(float)>>>(d_softmax_buffer.ptr, outer_size, inner_size);
    return ok(cudaGetLastError()) &&
           ok(cudaDeviceSynchronize()) &&
           ok(cudaMemcpy(data, d_softmax_buffer.ptr, bytes, cudaMemcpyDeviceToHost));
}

}  // namespace gpu_backend
}  // namespace nks_llm
