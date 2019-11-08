'''
(c) Marcelo Genanri 2019
Implementation of module that transforms a tensor to Block Floating Point (BFP).
This requires the number of bits for the mantissa, exponent, and the block size.
'''
import math
import torch
import torch.nn as nn

class BFPActivationLegacy(nn.Module):
    '''
    Activation module that transforms normal tensor to BFP with block size of blk
    This is legacy as this will (hopefully) run slower than theo's version.
    '''
    def __init__(self, mantissa, exponent, blk):
        super(BFPActivationLegacy, self).__init__()
        self.exp = exponent
        self.update_mantissa(mantissa)
        self.blk = blk
        self.max = 2**(self.exp-1)-1
        self.min = -2**(self.exp-1)

        self.__quantize__ = BFPQuant.apply

    def extra_repr(self):
        repr_str = ('exponent={exp}, mantissa={mts}, block_size={blk}')
        return repr_str.format(**self.__dict__)

    def update_mantissa(self, mantissa):
        self.mts = mantissa
        if self.mts is not None:
            self.min_m = -(2**self.mts)+1
            self.max_m = (2**self.mts)-1

    def forward(self, inp):
        # if bit is None, then use FP32
        if self.mts is None:
            return inp

        shp = inp.shape
        number_of_blocks = math.ceil(shp[1]/self.blk)
        pad_val = 0

        # Make sure that the tensor is a multiple of self.blk depthwise by adding a zero padding
        if shp[1] % self.blk != 0:
            pad_val = abs(shp[1]-number_of_blocks*self.blk)
            pad = torch.zeros(shp[0], pad_val, shp[2], shp[3])
            inp = torch.cat((inp, pad), dim=1)

        # Now we are sure that the inp tensor has a multiple of 32 in the depthwise axis
        inp = torch.unsqueeze(inp, 0)
        inp = torch.reshape(inp, (number_of_blocks, shp[0], self.blk, shp[2], shp[3]))

        inp = self.__quantize__(inp, self.min, self.max, self.mts, self.min_m, self.max_m)
        inp = torch.reshape(inp, (1, shp[0], shp[1]+pad_val, shp[2], shp[3]))

        inp = inp[0, :shp[0], :shp[1], ...]
        return inp


class BFPQuant(torch.autograd.Function):
    '''
    Activation function that transforms normal tensor to BFP with block size of blk
    This is also legacy as this will (hopefully) run slower than theo's version.
    Note that this is using the Straight-Through Operator for the backward function
    '''
    @staticmethod
    def __to_exponent_mantissa_width__(inp, max_log, mantissa_bitwidth, min_mantissa, max_mantissa):
        shp = inp.shape
        max_log = max_log.unsqueeze(2)
        # NOTE THAT THIS IS -1 BECAUSE OF THE LEADING 1 IN 1.b0b1b2b3...*2^E
        exponent_needed = (mantissa_bitwidth-max_log-1)*torch.ones(shp).to(inp.device)
        first_mant_w = torch.pow(2, exponent_needed)
        inp = inp*first_mant_w
        # Half LSB Rounding:
        inp = torch.round(inp)
        inp = torch.clamp(inp, min=min_mantissa, max=max_mantissa)
        inp = inp/first_mant_w
        return inp

    @staticmethod
    def __find_exponent__(inp, min_exponent, max_exponent):
        absolute = torch.abs(inp)
        value_log = torch.log2(absolute)
        value_log = torch.clamp(value_log, min_exponent, max_exponent)
        int_log = torch.floor(value_log)
        max_exponent, _ = torch.max(int_log, dim=2)
        return max_exponent

    @staticmethod
    def forward(ctx, inp, min_e, max_e, mantissa_bit, min_m, max_m):
        max_exponent = BFPQuant.__find_exponent__(inp, min_e, max_e)
        quantized_act = BFPQuant.__to_exponent_mantissa_width__(inp,
                                                                max_exponent,
                                                                mantissa_bit,
                                                                min_m,
                                                                max_m)
        return quantized_act

    @staticmethod
    def backward(ctx, grad):
        # STE Gradient
        return grad, None, None, None, None, None
