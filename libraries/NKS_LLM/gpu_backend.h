#ifndef NKS_LLM_GPU_BACKEND_H
#define NKS_LLM_GPU_BACKEND_H

#include <cstddef>

namespace nks_llm {
namespace gpu_backend {

bool is_available();
const char* backend_name();

bool matmul_2d(const float* a, const float* b, float* out,
               std::size_t m, std::size_t n, std::size_t p);

bool matmul_3d_2d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p);

bool matmul_3d_3d(const float* a, const float* b, float* out,
                  std::size_t batch, std::size_t m, std::size_t n, std::size_t p);

bool softmax_last_dim(float* data, std::size_t outer_size, std::size_t inner_size);

}  // namespace gpu_backend
}  // namespace nks_llm

#endif  // NKS_LLM_GPU_BACKEND_H
