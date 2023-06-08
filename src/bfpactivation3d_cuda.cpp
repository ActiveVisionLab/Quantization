// (c) Theo Costain 2021

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bfpactivation3d_cuda_forward(const torch::Tensor features,
                                                      const int32_t m_bits,
                                                      const int32_t block_size);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bfpactivation3d_forward(const torch::Tensor activations,
                                                 const int32_t m_bits, const int32_t block_size) {
    CHECK_INPUT(activations);

    return bfpactivation3d_cuda_forward(activations, m_bits, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bfpactivation3d_forward, "BFPActivation3D forward (CUDA)");
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
