# (c) Theo Costain 2019
"""PyTorch cuda code performing BFP Activation operation."""

import torch
from torch import nn
from torch.autograd import Function
import math

from . import bfpactivation_cpu
from . import bfpactivation_cuda


class BFPActivationFunctionGPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk, permute=True):
        # TODO permute activations to put C in last dim
        if permute:
            activations = activations.permute(0, 2, 3, 1).contiguous()
            outputs = bfpactivation_cuda.forward(activations, mantissa_bits, blk)

            output = outputs[0]
            output = output.permute(0, 3, 1, 2).contiguous()

        else:
            outputs = bfpactivation_cuda.forward(activations, mantissa_bits, blk)
            output = outputs[0]

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None


class BFPActivationFunctionCPU(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk, permute=True):
        if permute:
            activations = activations.permute(0, 2, 3, 1).contiguous()
            outputs = bfpactivation_cpu.forward(activations, mantissa_bits, blk)

            output = outputs[0]
            output = output.permute(0, 3, 1, 2).contiguous()
        else:
            outputs = bfpactivation_cpu.forward(activations, mantissa_bits, blk)
            output = outputs[0]
        # ctx.save_for_backward(output, argmax)

        return output

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None


class BFPActivationFunction(Function):
    @staticmethod
    def forward(ctx, activations, mantissa_bits, blk, permute=True):
        if activations.is_cuda:
            return BFPActivationFunctionGPU.apply(
                activations, mantissa_bits, blk, permute
            )
        elif not activations.is_cuda:
            return BFPActivationFunctionCPU.apply(
                activations, mantissa_bits, blk, permute
            )
        else:
            raise RuntimeError("All tensors not cuda or cpu tensors.")

    @staticmethod
    def backward(ctx, out_gradients):
        return out_gradients, None, None, None


class BFPActivation(nn.Module):
    def __init__(self, mantissa, blk, permute=True):
        super(BFPActivation, self).__init__()
        self.update_mantissa(mantissa)
        self.blk = blk
        self.bfp = BFPActivationFunction.apply
        self.permute = permute
        self.exp = 7
        self.max = 2 ** (self.exp - 1) - 1
        self.min = -2 ** (self.exp - 1)

    def update_mantissa(self, mantissa):
        self.mantissa = mantissa
        if self.mantissa is not None:
            self.min_m = -(2 ** self.mantissa) + 1
            self.max_m = (2 ** self.mantissa) - 1

    def extra_repr(self):
        repr_str = "exponent={exp}, mantissa={mantissa}, block_size={blk}"
        return repr_str.format(**self.__dict__)

    def forward(self, activations):
        return self.bfp(activations, self.mantissa, self.blk, self.permute)
