#include "gpu_backend.h"

namespace nks_llm {
namespace gpu_backend {

bool is_available() {
    return false;
}

const char* backend_name() {
    return "CPU";
}

bool matmul_2d(const float*, const float*, float*, std::size_t, std::size_t, std::size_t) {
    return false;
}

bool matmul_3d_2d(const float*, const float*, float*,
                  std::size_t, std::size_t, std::size_t, std::size_t) {
    return false;
}

bool matmul_3d_3d(const float*, const float*, float*,
                  std::size_t, std::size_t, std::size_t, std::size_t) {
    return false;
}

bool softmax_last_dim(float*, std::size_t, std::size_t) {
    return false;
}

}  // namespace gpu_backend
}  // namespace nks_llm
