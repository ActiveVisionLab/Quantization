// (c) Theo Costain 2021
#include <torch/extension.h>

#include <vector>

#define MAX_BLOCK_SIZE 64

#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000

template <typename scalar_t>
void forward(const torch::TensorAccessor<float, 5> activations, const uint32_t trunc_num,
             const uint32_t round_num, torch::TensorAccessor<float, 5> output, int32_t block_size) {
    const int32_t N = activations.size(0);
    const int32_t W = activations.size(1);
    const int32_t H = activations.size(2);
    const int32_t D = activations.size(3);
    const int32_t C = activations.size(4);

    // Faster integer division round up with overflow minimisation
    const int32_t blocks = 1 + ((C - 1) / block_size);

    for (int32_t n = 0; n < N; n++) {
        for (int32_t w = 0; w < W; w++) {
            for (int32_t h = 0; h < H; h++) {
                for (int32_t d = 0; d < D; d++) {
                    for (int32_t b = 0; b < blocks; b++) {
                        uint32_t max_e = 0;
                        uint32_t data[MAX_BLOCK_SIZE]; // Hardcoded limit to block size.
                        // max in neighborhood
                        for (int32_t c_block = 0; c_block < block_size; c_block++) {
                            int32_t c = c_block + b * block_size;
                            if (c >= C) {
                                continue;
                            }
                            std::memcpy(&data[c_block], &activations[n][w][h][d][c], sizeof(uint32_t));
                            uint32_t e = data[c_block] & EXP_MAGIC_NUM;
                            if (e > max_e) {
                                max_e = e;
                            }
                        }

                        for (int32_t c_block = 0; c_block < block_size; c_block++) {
                            int32_t c = c_block + b * block_size;
                            if (c >= C) {
                                continue;
                            }
                            // Extract the parts of the floating point number
                            uint32_t s = data[c_block] & SIG_MAGIC_NUM;
                            uint32_t e = data[c_block] & EXP_MAGIC_NUM;
                            uint32_t m = data[c_block] & MAN_MAGIC_NUM;

                            // calculate the required shift
                            uint32_t shift = (max_e - e) >> 23;

                            // convert into m form
                            uint32_t new_m = m | LEADING_1;

                            // shift the mantissa
                            new_m = new_m >> shift;

                            // round the value correctly (half LSB rounding)
                            new_m += round_num;

                            // correct if we round too far
                            if (new_m >> 24) {
                                new_m = m | LEADING_1;
                                new_m = new_m >> shift;
                            }

                            // truncate the mantissa
                            uint32_t trunc_m = new_m & trunc_num;

                            // build the quantised float
                            uint32_t out = s | max_e | trunc_m;

                            // put quantised float back into tensor
                            float f_out;
                            std::memcpy(&f_out, &out, sizeof out);

                            // correct back into 1+m form. This is because the calculations up to now
                            // have not taken into account the leading 1
                            if ((shift == 1) && (new_m >> 23 == 1)) {
                                // This block catches the error when the rouding does the mantissa
                                // correction for us.
                            } else if (shift != 0) {
                                f_out +=
                                    // TODO: Find another way of doing this:
                                    // The problem with shift is that it doesn't allow for decimal
                                    // points i.e. if max_e = -1, we would have to shift 1 >> 1,
                                    // which is always 0. s >> 31 ? 1 << ((max_e >> 23) - 127) : -(1
                                    // <<
                                    // ((max_e >> 23) - 127));
                                    s >> 31 ? pow(2, (((int32_t)(max_e >> 23)) - 127))
                                            : -pow(2, (((int32_t)(max_e >> 23))) - 127);
                            }
                            output[n][w][h][d][c] = f_out;
                        }
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> bfpactivation3d_forward(const torch::Tensor activations,
                                                 const int32_t m_bits, int32_t block_size) {

    auto output = torch::zeros_like(activations);

    // Generate some more magic numbers that cant be known at compile time.
    // The first is the truncation bitmask, and the second is a 1 in the LSB w.r.t.
    // trunction to round the values correctly
    const uint32_t trunc_num = (MAN_MAGIC_NUM >> (23 - (m_bits - 1))) << (23 - (m_bits - 1));
    const uint32_t round_num = ROUND_MAGIC_NUM >> (m_bits - 1);

    AT_DISPATCH_FLOATING_TYPES(activations.scalar_type(), "bfpactivation3d_forward_cpu", ([&] {
                                   forward<scalar_t>(activations.accessor<float, 5>(), trunc_num,
                                                     round_num, output.accessor<float, 5>(),
                                                     block_size);
                               }));

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bfpactivation3d_forward, "BFPActivation3D forward (CPU)");
}