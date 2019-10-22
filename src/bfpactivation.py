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
    def __init__(self, mantissa, exponent, blk):
        super(BFPActivation, self).__init__()
        self.exp = exponent
        self.mts = mantissa
        self.blk = blk
        self.max = 2**(self.exp-1)-1
        self.min = -2**(self.exp-1)
        if self.mts is not None:
            self.min_m = -(2**self.mts)+1
            self.max_m = (2**self.mts)-1

    def extra_repr(self):
        repr_str = ('exponent={exp}, mantissa={mts}, block_size={blk}')
        return repr_str.format(**self.__dict__)

    def forward(self, activations):
        # if bit is None, then use FP32
        if self.mts is None:
            return activations

        shp = activations.shape
        nmb_blocks = math.ceil(shp[1]/self.blk)
        pad_val = 0

        # Make sure that the tensor is a multiple of self.blk depthwise by adding a zero padding
        if shp[1] % self.blk != 0:
            pad_val = abs(shp[1]-nmb_blocks*self.blk)
            pad = torch.zeros(shp[0], pad_val, shp[2], shp[3])
            activations = torch.cat((activations, pad), dim=1)

        # Now we are sure that the inp tensor has a multiple of 32 in the depthwise axis
        activations = torch.unsqueeze(activations, 0)
        activations = torch.reshape(activations, (nmb_blocks, shp[0], self.blk, shp[2], shp[3]))

        if activations.is_cuda:
            activations = BFPActivationFunctionGPU.apply(activations, self.mts)
        elif not activations.is_cuda:
            activations = BFPActivationFunctionCPU.apply(activations, self.mts)
        else:
            raise RuntimeError("All tensors not cuda or cpu tensors.")

        activations = torch.reshape(activations, (1, shp[0], shp[1]+pad_val, shp[2], shp[3]))

        activations = activations[0, :shp[0], :shp[1], ...]
        return activations
