// (c) Theo Costain 2020
#include <bitset>
#include <iostream>
#include <torch/extension.h>

#include <vector>

#define MAX_BLOCK_SIZE 64

#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000

std::ostream &print_bits(std::ostream &outstream, std::bitset<32> &a) {
    outstream << a[31] << " ";
    for (int i = 30; i > 22; i--) {
        outstream << a[i];
    }
    outstream << " ";
    for (int i = 22; i >= 0; i--) {
        outstream << a[i];
    }
    outstream << std::endl;
    return outstream;
}

template <typename scalar_t>
void forward(const torch::TensorAccessor<float, 4> activations, const uint32_t trunc_num,
             const uint32_t round_num, torch::TensorAccessor<float, 4> output, int32_t block_size) {
    const int32_t N = activations.size(0);
    const int32_t W = activations.size(1);
    const int32_t H = activations.size(2);
    const int32_t C = activations.size(3);

    // Faster integer division round up with overflow minimisation
    const int32_t blocks = 1 + ((C - 1) / block_size);

    for (int32_t n = 0; n < N; n++) {
        for (int32_t w = 0; w < W; w++) {
            for (int32_t h = 0; h < H; h++) {
                for (int32_t b = 0; b < blocks; b++) {
                    uint32_t max_e = 0;
                    uint32_t data[MAX_BLOCK_SIZE]; // Hardcoded limit to block size.
                    // max in neighborhood
                    for (int32_t c_block = 0; c_block < block_size; c_block++) {
                        int32_t c = c_block + b * block_size;
                        if (c >= C) {
                            continue;
                        }
                        std::memcpy(&data[c_block], &activations[n][w][h][c], sizeof(uint32_t));
                        uint32_t e = data[c_block] & EXP_MAGIC_NUM;
                        if (e > max_e) {
                            max_e = e;
                        }
                    }

                    /**** For Debugging ******/
                    if (n == 0 && w == 0 && h == 0) {
                        std::bitset<32> bitset_max_e = max_e;
                        std::cout << "Max Exponent         ";
                        print_bits(std::cout, bitset_max_e);
                        std::cout << std::endl;
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

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_orig = data[c_block];
                            std::cout << "Original bits        ";
                            print_bits(std::cout, bitset_orig);
                            std::bitset<32> bitset_sign = s;
                            std::cout << "Sign bitset          ";
                            print_bits(std::cout, bitset_sign);
                            std::bitset<32> bitset_exponent = e;
                            std::cout << "Exponent bitset      ";
                            print_bits(std::cout, bitset_exponent);
                            std::bitset<32> bitset_mantissa = m;
                            std::cout << "Mantissa bitset      ";
                            print_bits(std::cout, bitset_mantissa);
                        }

                        // calculate the required shift
                        uint32_t shift = (max_e - e) >> 23;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::cout << "Shift Necessary      " << shift << std::endl;
                        }

                        // convert into m form
                        uint32_t new_m = m | LEADING_1;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_new_m(new_m);
                            std::cout << "Mantissa bits w/ l1  ";
                            print_bits(std::cout, bitset_new_m);
                        }

                        // shift the mantissa
                        new_m = new_m >> shift;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_new_m(new_m);
                            std::cout << "Mantissa shifted     ";
                            print_bits(std::cout, bitset_new_m);
                        }

                        // round the value correctly (half LSB rounding)
                        new_m += round_num;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_new_m(new_m);
                            std::cout << "Mantissa rounded     ";
                            print_bits(std::cout, bitset_new_m);
                        }

                        // correct if we round too far
                        if (new_m >> 24) {
                            new_m = m | LEADING_1;
                            new_m = new_m >> shift;

                            /**** For Debugging ******/
                            if (n == 0 && w == 0 && h == 0) {
                                std::bitset<32> bitset_new_m(new_m);
                                std::cout << "Mantissa corrected     ";
                                print_bits(std::cout, bitset_new_m);
                            }
                        }

                        // truncate the mantissa
                        uint32_t trunc_m = new_m & trunc_num;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_new_m(trunc_m);
                            std::cout << "Mantissa truncated   ";
                            print_bits(std::cout, bitset_new_m);
                        }

                        // build the quantised float
                        uint32_t out = s | max_e | trunc_m;

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::bitset<32> bitset_out(out);
                            std::cout << "Quantized Float      ";
                            print_bits(std::cout, bitset_out);
                        }

                        // put quantised float back into tensor
                        float f_out;
                        std::memcpy(&f_out, &out, sizeof out);

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::cout << "Quant float format   " << f_out << std::endl;
                            std::cout << "Quant float minus l1 "
                                      << f_out + (((s >> 31) - 0.5) * 2) *
                                                     pow(2, (int32_t)(max_e >> 23) - 127)
                                      << std::endl;
                        }

                        // correct back into 1+m form. This is because the calculations up to now
                        // have not taken into account the leading 1
                        if ((shift == 1) && (new_m >> 23 == 1)) {
                            // This block catches the error when the rouding does the mantissa
                            // correction for us.
                            output[n][w][h][c] = f_out;
                            continue;
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

                        /**** For Debugging ******/
                        if (n == 0 && w == 0 && h == 0) {
                            std::cout << "Quant float final    " << f_out << std::endl << std::endl;
                        }
                        output[n][w][h][c] = f_out;
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> bfpactivation_forward(const torch::Tensor activations,
                                                 const int32_t m_bits, int32_t block_size) {

    auto output = torch::zeros_like(activations);

    // Generate some more magic numbers that cant be known at compile time.
    // The first is the truncation bitmask, and the second is a 1 in the LSB w.r.t.
    // trunction to round the values correctly
    const uint32_t trunc_num = (MAN_MAGIC_NUM >> (23 - (m_bits - 1))) << (23 - (m_bits - 1));
    const uint32_t round_num = ROUND_MAGIC_NUM >> (m_bits - 1);

    AT_DISPATCH_FLOATING_TYPES(activations.scalar_type(), "bfpactivation_forward_cpu", ([&] {
                                   forward<scalar_t>(activations.accessor<float, 4>(), trunc_num,
                                                     round_num, output.accessor<float, 4>(),
                                                     block_size);
                               }));

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bfpactivation_forward, "BFPActivation forward (CPU)");
}