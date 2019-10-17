# (c) Theo Costain 2019
"""Tensorflow op performing flex convolution operation."""

from torch import nn
from torch.autograd import Function

from . import bfpactivation_cpu


# class FlexPoolFunctionGPU(Function):
#     @staticmethod
#     def forward(ctx, features, neighborhood):
#         outputs = flexpool_cuda.forward(features, neighborhood.int())

#         output, argmax = outputs[:2]
#         ctx.save_for_backward(output, argmax)

#         return output

#     @staticmethod
#     def backward(ctx, out_gradients):
#         output, argmax = ctx.saved_tensors

#         outputs = flexpool_cuda.backward(argmax.int(), out_gradients)
#         gradients = outputs[0]

#         return gradients, None


class BFPActivationFunction(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits=3):
        outputs = bfpactivation_cpu.forward(activations, mantissa_bits, 0)

        output = outputs[0]
        # ctx.save_for_backward(output, argmax)

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None


class FlexPool(nn.Module):
    def __init__(self, mantissa_bits=3):
        super(FlexPool, self).__init__()
        self.mantissa_bits = mantissa_bits

    def forward(self, activations):
        if activations.is_cuda:
            raise RuntimeError("CUDA version not implemented yet")
        elif not activations.is_cuda:
            return BFPActivationFunction.apply(activations, self.mantissa_bits)
        else:
            raise RuntimeError("All tensors not cuda or cpu tensors.")
