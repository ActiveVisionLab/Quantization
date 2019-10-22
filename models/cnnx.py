'''
(c) Marcelo Genanri 2019
Implementation of cnnx (test cnn) to train Block Floating Point (BFP) and DSConv
using CIFAR10 dataset.
'''
import torch
import torch.nn.functional as F

from DSConv.nn.dsconv2d import DSConv2d
from DSConv.nn.bfp_quantization import BFPActivationLegacy as BFPActivation
from DSConv.nn.quantized_module import QuantizedModule

class BaseConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel, block_size, bit, **kwargs):
        super(BaseConv, self).__init__()
        self.activation = BFPActivation(bit, 7, block_size)
        self.conv = DSConv2d(in_planes, out_planes, kernel, block_size, bit, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return F.relu(self.bn(self.conv(self.activation(x))))


class CNNX(QuantizedModule):
    '''
    Toy module to test DSConv in the CIFAR10 dataset.
    Consists of X convolutions followed by BFPActivation functions
    with Relu() and BatchNorms at the end of each
    '''
    def __init__(self, bits):
        super(CNNX, self).__init__(bits, self.number_bits)
        self.block_size = 32

        bit = self.bits.pop(0)
        self.conv1 = DSConv2d(3, 64, (3, 3), block_size=32, bit=bit, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.features1, outch = self.__make_layers__(64, 3)
        self.max_pool1 = torch.nn.MaxPool2d(2, stride=2)

        self.features2, outch = self.__make_layers__(outch, 3)
        self.max_pool2 = torch.nn.MaxPool2d(2, stride=2)

        self.features3, outch = self.__make_layers__(outch, 3)
        self.avg_pool = torch.nn.AvgPool2d(8)

        self.linear = torch.nn.Linear(outch, 10)

    def __make_layers__(self, initial_channel, expansion):
        number_layers = int((self.number_bits-1)/3)
        layers = []
        for i in range(number_layers):
            inch = initial_channel*(2**(i//expansion))
            outch = initial_channel*(2**((i+1)//expansion))
            bit = self.bits.pop(0)
            layers.append(BaseConv(inch, outch, (3, 3), self.block_size, bit, padding=1))

        return torch.nn.Sequential(*layers), outch

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features1(x)
        x = self.max_pool1(x)
        x = self.features2(x)
        x = self.max_pool2(x)
        x = self.features3(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class CNN10(CNNX):
    ''' Defines CNNX for 10 layers '''
    number_bits = 10
