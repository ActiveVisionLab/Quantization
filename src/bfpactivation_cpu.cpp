// (c) Theo Costain 2019
#include <torch/extension.h>

#include <vector>

#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define SIG_MAGIC_NUM 0x80000000

template <typename scalar_t>
void forward(const torch::TensorAccessor<float, 5> activations, const int32_t m_bits,
             const int32_t e_bits, torch::TensorAccessor<float, 5> output) {
    const int32_t B = activations.size(0);
    const int32_t N = activations.size(1);
    const int32_t C = activations.size(2);
    const int32_t H = activations.size(3);
    const int32_t W = activations.size(4);

    for (int32_t b = 0; b < B; b++) {
        for (int32_t n = 0; n < N; n++) {
            for (int32_t w = 0; w < W; w++) {
                for (int32_t h = 0; h < H; h++) {
                    int32_t max_e = 0;
                    // max in neighborhood
                    for (int32_t c = 0; c < C; c++) {
                        int32_t data;
                        std::memcpy(&data, &activations[b][n][c][w][h], sizeof data);
                        int32_t e = data & EXP_MAGIC_NUM;
                        if (e > max_e) {
                            max_e = e;
                        }
                    }
                    for (int32_t c = 0; c < C; c++) {
                        // Load data from tensor
                        int32_t data;
                        std::memcpy(&data, &activations[b][n][c][w][h], sizeof data);
                        // Extract exponent from the data
                        int32_t e = data & EXP_MAGIC_NUM;
                        // Extract mantissa from data and convert to 1+m form
                        int32_t m = data & MAN_MAGIC_NUM | 0x800000;
                        // Compute difference in exponents
                        int32_t diff = max_e - e;

                        int32_t new_m = m;
                        int32_t new_e = max_e;
                        // For non-zero diff rightshift by the difference and correct the exponent
                        if (diff != 0) {
                            new_m = m >> diff;
                            new_e = ((new_e >> 23) - diff) << 23;
                        }
                        // Truncate and reconvert to m form
                        new_m = new_m & 0x00600000;
                        int32_t new_data = (SIG_MAGIC_NUM & data) | max_e | new_m;
                        std::memcpy(&output[b][n][c][w][h], &new_data, sizeof(float));
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> bfpactivation_forward(const torch::Tensor activations,
                                                 const int32_t m_bits, const int32_t e_bits) {

    auto output = torch::zeros_like(activations);

    AT_DISPATCH_FLOATING_TYPES(activations.type(), "bfpactivation_forward_cpu", ([&] {
                                   forward<scalar_t>(activations.accessor<float, 5>(), m_bits,
                                                     e_bits, output.accessor<float, 5>());
                               }));

    return {output};
}

// template <typename scalar_t, typename scalar_int_t>
// void backward(const torch::TensorAccessor<scalar_int_t, 3> argmax,
//               const torch::TensorAccessor<scalar_t, 3> out_gradients,
//               torch::TensorAccessor<scalar_t, 3> gradients)
// {
// }

// std::vector<torch::Tensor> bfpactivation_backward(const torch::Tensor argmax,
//                                                   const torch::Tensor out_gradients)
// {
//     auto gradients = torch::zeros_like(out_gradients);

//     AT_DISPATCH_FLOATING_TYPES(out_gradients.type(), "bfpactivation_backward_cpu", ([&] {
//                                    backward<scalar_t>(argmax.accessor<int32_t, 3>(),
//                                                       out_gradients.accessor<scalar_t, 3>(),
//                                                       gradients.accessor<scalar_t, 3>());
//                                }));

//     return {gradients};
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bfpactivation_forward, "BFPActivation forward (CPU)");
    // m.def("backward", &bfpactivation_backward, "BFPActivation backward (CPU)");
}