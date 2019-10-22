// (c) Theo Costain 2019

// This was not compiling with extension.h on anubis5 with cuda 9. This is supposed to fix it.
#include <torch/types.h>
// #include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000

inline int up2(int len, int th) { return (len - 1) / th + 1; };

namespace {
template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits, size_t> activations,
    torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits, size_t> output,
    const int32_t m_bits, const uint32_t trunc_num, const uint32_t round_num) {

    const int32_t w = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t h = threadIdx.y + blockIdx.y * blockDim.y;
    const int32_t b = blockIdx.z;

    const int32_t N = activations.size(0);
    const int32_t B = activations.size(1);
    const int32_t C = activations.size(2);
    const int32_t H = activations.size(3);
    const int32_t W = activations.size(4);

    if ((w >= W) | (h >= H) | (b >= B)) {
        return;
    }

    for (int32_t n = 0; n < N; n++) {
        uint32_t max_e = 0;
        // max in neighborhood
        for (int32_t c = 0; c < C; c++) {
            uint32_t data;
            memcpy(&data, &activations[n][b][c][w][h], sizeof data);
            uint32_t e = data & EXP_MAGIC_NUM;
            if (e > max_e) {
                max_e = e;
            }
        }
        for (int32_t c = 0; c < C; c++) {
            // Load data from tensor
            uint32_t data;
            memcpy(&data, &activations[n][b][c][w][h], sizeof data);

            // Extract the parts of the floating point number
            uint32_t s = data & SIG_MAGIC_NUM;
            uint32_t e = data & EXP_MAGIC_NUM;
            uint32_t m = data & MAN_MAGIC_NUM;

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
            memcpy(&output[n][b][c][w][h], &out, sizeof out);

            // correct back into 1+m form.
            if ((shift == 1) && (new_m >> 23 == 1)) {
                // This block catches the error when the rouding does the mantissa
                // correction for us.
                continue;
            } else if (shift != 0) {
                output[n][b][c][w][h] +=
                    // TODO: Find another way of doing this:
                    // The problem with shift is that it doesn't allow for decimal
                    // points i.e. if max_e = -1, we would have to shift 1 >> 1,
                    // which is always 0.
                    // and pow is a "slow" operation
                    s >> 31 ? pow(2, (((int32_t)(max_e >> 23)) - 127))
                            : -pow(2, (((int32_t)(max_e >> 23))) - 127);
            }
        }
    }
}
} // namespace

std::vector<at::Tensor> bfpactivation_cuda_forward(const torch::Tensor activations,
                                                   const int32_t m_bits) {

    const int32_t N = activations.size(0);
    const int32_t B = activations.size(1);
    const int32_t C = activations.size(2);
    const int32_t H = activations.size(3);
    const int32_t W = activations.size(4);

    auto output = torch::zeros({N, B, C, H, W},
                               torch::dtype(activations.dtype()).device(activations.device()));

    // Generate some more magic numbers that cant be known at compile time.
    // The first is the truncation bitmask, and the second is a 1 in the LSB w.r.t.
    // trunction to round the values correctly
    const uint32_t trunc_num = (MAN_MAGIC_NUM >> (23 - (m_bits - 1))) << (23 - (m_bits - 1));
    const uint32_t round_num = ROUND_MAGIC_NUM >> (m_bits - 1);

    const int32_t threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(W, threads), up2(H, threads), B);

    AT_DISPATCH_FLOATING_TYPES(
        activations.type(), "bfpactivation_forward_cuda", ([&] {
            forward_kernel<scalar_t><<<grid, block>>>(
                activations.packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(),
                output.packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(), m_bits,
                trunc_num, round_num);
        }));

    return {output};
}
