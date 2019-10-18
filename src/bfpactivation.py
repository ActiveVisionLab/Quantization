# (c) Theo Costain 2019
"""Tensorflow op performing flex convolution operation."""

from torch import nn
from torch.autograd import Function

from . import bfpactivation_cpu
from . import bfpactivation_cuda


class BFPActivationFunctionGPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits):
        outputs = bfpactivation_cuda.forward(activations, mantissa_bits)

        output = outputs[0]

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None


class BFPActivationFunctionCPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits=3):
        outputs = bfpactivation_cpu.forward(activations, mantissa_bits)

        output = outputs[0]
        # ctx.save_for_backward(output, argmax)

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None


class BFPActivation(nn.Module):
    def __init__(self, mantissa_bits=3):
        super(BFPActivation, self).__init__()
        self.mantissa_bits = mantissa_bits

    def forward(self, activations):
        if activations.is_cuda:
            pass
            # return BFPActivationFunctionGPU.apply(activations, self.mantissa_bits)
        elif not activations.is_cuda:
            return BFPActivationFunctionCPU.apply(activations, self.mantissa_bits)
        else:
            raise RuntimeError("All tensors not cuda or cpu tensors.")
