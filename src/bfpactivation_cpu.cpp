// (c) Theo Costain 2019
#include <torch/extension.h>

#include <vector>

#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000

template <typename scalar_t>
void forward(const torch::TensorAccessor<float, 5> activations, const uint32_t trunc_num, const uint32_t round_num,
             const int32_t e_bits, torch::TensorAccessor<float, 5> output) {
    const int32_t N = activations.size(0);
    const int32_t B = activations.size(1);
    const int32_t C = activations.size(2);
    const int32_t H = activations.size(3);
    const int32_t W = activations.size(4);

    for (int32_t b = 0; b < B; b++) {
        for (int32_t n = 0; n < N; n++) {
            for (int32_t w = 0; w < W; w++) {
                for (int32_t h = 0; h < H; h++) {
                    uint32_t max_e = 0;
                    // max in neighborhood
                    for (int32_t c = 0; c < C; c++) {
                        uint32_t data;
                        std::memcpy(&data, &activations[n][b][c][w][h], sizeof data);
                        uint32_t e = data & EXP_MAGIC_NUM;
                        // std::cout << e << std::endl;
                        if (e > max_e) {
                            max_e = e;
                        }
                    }
                    for (int32_t c = 0; c < C; c++) {
                        // Load data from tensor
                        uint32_t data;
                        std::memcpy(&data, &activations[n][b][c][w][h], sizeof data);

                        // Extract the parts of the floating point number
                        uint32_t s = data & SIG_MAGIC_NUM;
                        uint32_t e = data & EXP_MAGIC_NUM;
                        uint32_t m = data & MAN_MAGIC_NUM;
                        // std::cout << "s:" << std::hex << (s >> 31) << " e:" << std::hex << e
                        //           << " m:" << std::hex << m << std::endl;

                        // calculate the required shift
                        uint32_t shift = (max_e - e) >> 23;
                        // std::cout << "shift:" << shift << std::endl;

                        // convert into m form 
                        uint32_t new_m = m | LEADING_1;

                        //shift the mantissa
                        new_m = new_m >> shift;

                        // round the value correctly (half LSB rounding)
                        new_m += round_num;

                        // correct if we round too far
                        // if ((new_m >> 24) > 1){
                        //     // new_m -= round_num;
                        //     new_m = m | LEADING_1;
                        //     new_m = new_m >> shift;
                        // }
                        
                        // truncate the mantissa
                        uint32_t trunc_m = new_m & trunc_num;

                        // build the quantised float
                        uint32_t out = s | max_e | trunc_m;

                        // put quantised float back into tensor
                        std::memcpy(&output[n][b][c][w][h], &out, sizeof out);

                        // std::cout << "Before correcting the 1+m form " << output[n][b][c][w][h] << std::endl;

                        // std::cout << (s>>31) << std::endl;

                        // correct back into 1+m form.
                        if(shift != 0){
                            output[n][b][c][w][h] +=
                            // TODO: Find another way of doing this:
                            // The problem with shift is that it doesn't allow for decimal points
                            // i.e. if max_e = -1, we would have to shift 1 >> 1, which is always 0.
                                // s >> 31 ? 1 << ((max_e >> 23) - 127) : -(1 << ((max_e >> 23) - 127));
                                s >> 31 ? pow(2, ((max_e >> 23) - 127)) : - pow(2, ((max_e >> 23))-127);
                        }

                    }

                    // for (int32_t c = 0; c < C; c++) {
                    //     // Load data from tensor
                    //     uint32_t data;
                    //     std::memcpy(&data, &activations[n][b][c][w][h], sizeof data);
                    //     // Extract exponent from the data
                    //     uint32_t e = data & EXP_MAGIC_NUM;
                    //     // Extract mantissa from data and convert to 1+m form
                    //     uint32_t m = data & MAN_MAGIC_NUM | 0x800000;
                    //     // Compute difference in exponents
                    //     uint32_t diff = (max_e - e) >> 23;

                    //     uint32_t new_m = m;
                    //     uint32_t new_e = max_e;
                    //     // For non-zero diff rightshift by the difference and correct the
                    //     exponent if (diff != 0) {
                    //         new_m = m >> diff;
                    //         new_e = ((new_e >> 23) - diff) << 23;
                    //     }
                    //     // Truncate and reconvert to m form
                    //     new_m = new_m & 0x00600000;
                    //     uint32_t new_data = (SIG_MAGIC_NUM & data) | max_e | new_m;
                    //     std::memcpy(&output[n][b][c][w][h], &new_data, sizeof(float));
                    //     // if (diff!=0){
                    //     //     output[n][b][c][w][h] -= (2 << ((max_e >> 23)-127));
                    //     //     std::cout << (max_e>>23) << " " << (2<< ((max_e >> 23)-127)) <<
                    //     //     std::endl;
                    //     // }
                    // }
                }
            }
        }
    }
}

std::vector<torch::Tensor> bfpactivation_forward(const torch::Tensor activations,
                                                 const int32_t m_bits, const int32_t e_bits) {

    auto output = torch::zeros_like(activations);
    //TODO comments for these two lines
    const uint32_t trunc_num = (MAN_MAGIC_NUM >> (23 - (m_bits - 1))) << (23 - (m_bits - 1));
    const uint32_t round_num = ROUND_MAGIC_NUM >> (m_bits - 1);
    // std::cout << std::hex << trunc_num << std::endl;
    // std::cout << std::hex << round_num << std::endl;

    AT_DISPATCH_FLOATING_TYPES(activations.type(), "bfpactivation_forward_cpu", ([&] {
                                   forward<scalar_t>(activations.accessor<float, 5>(), trunc_num, round_num,
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