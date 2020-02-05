// (c) Theo Costain 2020

// This was not compiling with extension.h on anubis5 with cuda 9. This is supposed to fix it.
#include <torch/types.h>
// #include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <vector>

#define gpuErrchk(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000

namespace {
template <typename scalar_t, int block_size>
__global__ void
forward_kernel(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> activations,
               torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output,
               const int32_t m_bits, const uint32_t trunc_num, const uint32_t round_num) {

    const int32_t w = blockIdx.x;
    const int32_t h = blockIdx.y;
    const int32_t n = blockIdx.z;

    const int32_t c_block = threadIdx.x;

    typedef cub::BlockReduce<uint32_t, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ uint32_t max_e;

    const int32_t N = activations.size(0);
    const int32_t W = activations.size(1);
    const int32_t H = activations.size(2);
    const int32_t C = activations.size(3);

    // Faster integer division round up with overflow minimisation
    const int32_t blocks = 1 + ((C - 1) / block_size);

    if ((w >= W) | (h >= H) | (n >= N)) {
        return;
    }

    for (int32_t b = 0; b < blocks; b++) {

        int32_t c = c_block + b * block_size;

        uint32_t data = 0;
        if (c < C) {
            // load data from gloabal memory
            memcpy(&data, &activations[n][w][h][c], sizeof(uint32_t));
        }

        uint32_t e = data & EXP_MAGIC_NUM;

        // Block reduce max
        uint32_t result = BlockReduce(temp_storage).Reduce(e, cub::Max());

        if (threadIdx.x == 0) {
            max_e = result;
        }

        // TODO: investigate whether loading max_e into a thread register rather
        // than reading from shared mem every time we need it is faster.

        __syncthreads();

        if (c < C) {
            // Extract the parts of the floating point number
            uint32_t s = data & SIG_MAGIC_NUM;
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
            float f_out;
            memcpy(&f_out, &out, sizeof out);

            // correct back into 1+m form.
            if ((shift == 1) && (new_m >> 23 == 1)) {
                // This block catches the error when the rouding does the mantissa
                // correction for us.
            } else if (shift != 0) {
                f_out +=
                    // TODO: Find another way of doing this:
                    // The problem with shift is that it doesn't allow for decimal
                    // points i.e. if max_e = -1, we would have to shift 1 >> 1,
                    // which is always 0.
                    // and pow is a "slow" operation
                    s >> 31 ? pow(2, (((int32_t)(max_e >> 23)) - 127))
                            : -pow(2, (((int32_t)(max_e >> 23))) - 127);
            }
            output[n][w][h][c] = f_out;
        }
    }
}
} // namespace

std::vector<at::Tensor> bfpactivation_cuda_forward(const torch::Tensor activations,
                                                   const int32_t m_bits, const int32_t block_size) {

    const int32_t N = activations.size(0);
    const int32_t W = activations.size(1);
    const int32_t H = activations.size(2);
    const int32_t C = activations.size(3);

    auto output =
        torch::zeros({N, W, H, C}, torch::dtype(activations.dtype()).device(activations.device()));

    // Generate some more magic numbers that cant be known at compile time.
    // The first is the truncation bitmask, and the second is a 1 in the LSB w.r.t.
    // trunction to round the values correctly
    const uint32_t trunc_num = (MAN_MAGIC_NUM >> (23 - (m_bits - 1))) << (23 - (m_bits - 1));
    const uint32_t round_num = ROUND_MAGIC_NUM >> (m_bits - 1);

    const int32_t threads = block_size;
    dim3 block(threads);
    dim3 grid(W, H, N);

    switch (block_size) {
    case 16:
        AT_DISPATCH_FLOATING_TYPES(
            activations.scalar_type(), "bfpactivation_forward_cuda", ([&] {
                forward_kernel<scalar_t, 16><<<grid, block>>>(
                    activations.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), m_bits,
                    trunc_num, round_num);
            }));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        break;
    case 32:
        AT_DISPATCH_FLOATING_TYPES(
            activations.scalar_type(), "bfpactivation_forward_cuda", ([&] {
                forward_kernel<scalar_t, 32><<<grid, block>>>(
                    activations.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), m_bits,
                    trunc_num, round_num);
            }));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        break;
    case 64:
        AT_DISPATCH_FLOATING_TYPES(
            activations.scalar_type(), "bfpactivation_forward_cuda", ([&] {
                forward_kernel<scalar_t, 64><<<grid, block>>>(
                    activations.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(), m_bits,
                    trunc_num, round_num);
            }));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        break;
    default:
        throw "Unsupported Block Size for quantisation!";
    }

    return {output};
}
