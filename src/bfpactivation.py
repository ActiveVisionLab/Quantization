# (c) Theo Costain 2020
"""Tensorflow op performing flex convolution operation."""

from torch import nn
from torch.autograd import Function

from . import bfpactivation_cpu
from . import bfpactivation_cuda


class BFPActivationFunctionGPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk):
        # TODO permute activations to put C in last dim
        activations = activations.permute(0, 2, 3, 1).contiguous()
        outputs = bfpactivation_cuda.forward(activations, mantissa_bits, blk)

        output = outputs[0]
        output = output.permute(0, 3, 1, 2).contiguous()

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None


class BFPActivationFunctionCPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk):
        activations = activations.permute(0, 2, 3, 1).contiguous()
        outputs = bfpactivation_cpu.forward(activations, mantissa_bits, blk)

        output = outputs[0]
        output = output.permute(0, 3, 1, 2).contiguous()
        # ctx.save_for_backward(output, argmax)

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None


class BFPActivationFunction(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk):
        if activations.is_cuda:
            return BFPActivationFunctionGPU.apply(activations, mantissa_bits, blk)
        elif not activations.is_cuda:
            return BFPActivationFunctionCPU.apply(activations, mantissa_bits, blk)
        else:
            raise RuntimeError("All tensors not cuda or cpu tensors.")

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None


class BFPActivation(nn.Module):
    def __init__(self, mantissa_bits, blk):
        super(BFPActivation, self).__init__()
        self.mantissa_bits = mantissa_bits
        self.blk = blk
        self.bfp = BFPActivationFunction.apply

    def forward(self, activations):
        return self.bfp(activations, self.mantissa_bits, self.blk)
