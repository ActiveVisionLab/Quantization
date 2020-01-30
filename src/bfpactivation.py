# (c) Theo Costain 2020
"""Tensorflow op performing flex convolution operation."""

from torch import nn
from torch.autograd import Function

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
        return out_gradients, None, None


class BFPActivation(nn.Module):
    def __init__(self, mantissa_bits, blk, permute=True):
        super(BFPActivation, self).__init__()
        self.mantissa_bits = mantissa_bits
        self.blk = blk
        self.bfp = BFPActivationFunction.apply
        self.permute = permute

    def forward(self, activations):
        return self.bfp(activations, self.mantissa_bits, self.blk, self.permute)
