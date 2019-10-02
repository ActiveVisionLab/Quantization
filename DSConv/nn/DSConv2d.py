# Other
import math

# PyTorch
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd, Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from .Quantization import DSConvQuant

class DSConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, block_size, bit, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(DSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                       dilation, False, _pair(0), groups, bias, padding_mode)
        self.nmb_blk = math.ceil(((in_channels)/(block_size*groups)))
        self.blk = block_size

        # If bit is None, then the layer should be FP32
        self.bit = bit
        if bit is not None:
            self.minV = -1*pow(2, bit-1)
            self.maxV = pow(2, bit-1)-1

        self.alpha = torch.Tensor(out_channels, self.nmb_blk, *kernel_size)
        self.intw = torch.Tensor(self.weight.size())
        self.quant_w = torch.Tensor(out_channels, in_channels, *kernel_size)

        self.__quantize__ = DSConvQuant.apply

    def extra_repr(self):
        st = super(DSConv2d, self).extra_repr() + ', block_size={blk}, bit={bit}'
        return st.format(**self.__dict__)

    def quantize(self):
        self.quant_w, self.intw, self.alpha = self.__quantize__(self.weight, self.blk, self.bit, self.nmb_blk)


    def forward(self, input):
        self.quant_w = self.quant_w.to(self.weight.device)
        return F.conv2d(input, self.quant_w, self.bias, self.stride, self.padding, self.dilation, self.groups)

