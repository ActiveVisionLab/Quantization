"""
(c) Marcelo Genanri 2019
Implementation of DSConv algorithm.
The Conv2d modules in a neural network should be simply substituted with this
module and it should work just fine (provided right arguments)
"""
# Other
import math

# PyTorch
import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from .dsconv_quantization import DSConvQuant


class DSConv3d(_ConvNd):
    """
    The difference between this and the Conv3d module is the bit and block_size arguments.
    In order to be used properly, the functio quantize() should be called,
    which transfors the fp32 weight into DSConv style according to bit and block_size args.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bit: int,
        block_size=32,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
        )
        self.nmb_blk = math.ceil(((in_channels) / (block_size * groups)))
        self.groups = groups
        self.blk = block_size

        # If bit is None, then the layer should be FP32
        self.update_bit(bit)

        self.alpha = torch.Tensor(out_channels, self.nmb_blk, *kernel_size)
        self.intw = torch.Tensor(self.weight.size())
        self.quant_w = torch.Tensor(out_channels, in_channels // groups, *kernel_size)

        self.__quantize__ = DSConvQuant.apply

    def extra_repr(self):
        repr_str = super().extra_repr() + ", block_size={blk}, bit={bit}"
        return repr_str.format(**self.__dict__)

    def update_bit(self, bit):
        self.bit = bit
        if bit is not None:
            self.min_v = -1 * pow(2, bit - 1)
            self.max_v = pow(2, bit - 1) - 1

    def quantize(self):
        """Transforms fp32 weights to quant_w in DSConv style"""
        (self.quant_w, self.intw, self.alpha) = self.__quantize__(
            self.weight, self.blk, self.bit, self.nmb_blk
        )

    def forward(self, inp):
        self.quant_w = self.quant_w.to(self.weight.device)
        return F.conv3d(
            inp,
            self.quant_w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
