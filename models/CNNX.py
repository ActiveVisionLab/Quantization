from DSConv.nn.DSConv2d import DSConv2d
from DSConv.nn.Activation import BFPActivation
from DSConv.nn.QuantizedModule import QuantizedModule
import torch
import torch.nn.functional as F

class BaseConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel, block_size, bit, **kwargs):
        super(BaseConv, self).__init__()
        self.activation = BFPActivation(bit, 7, block_size)
        self.conv = DSConv2d(in_planes, out_planes, kernel, block_size, bit, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return F.relu(self.bn(self.conv(self.activation(x))))


class CNNX(QuantizedModule):

    def __init__(self, bits):
        super(CNNX, self).__init__(bits, self.number_bits)
        self.block_size=32

        bit = self.bits.pop(0)
        self.conv1 = DSConv2d(3, 64, (3, 3), block_size=32, bit=bit, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.features1, outch  = self.make_layers(64, 3)
        self.max_pool1 = torch.nn.MaxPool2d(2, stride=2)

        self.features2, outch = self.make_layers(outch, 3)
        self.max_pool2 = torch.nn.MaxPool2d(2, stride=2)

        self.features3, outch = self.make_layers(outch, 3)
        self.avg_pool = torch.nn.AvgPool2d(8)

        self.linear = torch.nn.Linear(outch, 10)

    def make_layers(self, initial_channel, expansion):
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
    number_bits = 10

    def __init__(self, bits):
        super(CNN10, self).__init__(bits)

