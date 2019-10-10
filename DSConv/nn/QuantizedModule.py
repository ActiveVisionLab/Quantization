import torch
from DSConv.nn.DSConv2d import DSConv2d


class QuantizedModule(torch.nn.Module):
    def __init__(self, bits, number_bits):
        super(QuantizedModule, self).__init__()
        if type(bits) == int or bits == None:
            self.bits = [bits for _ in range(number_bits)]
        else:
            assert(number_bits == len(bits))
            self.bits = bits

    def quantize(self):
        [m.quantize() for m in self.modules() if isinstance(m, DSConv2d)]

