from DSConv.nn.DSConv2d import DSConv2d
from DSConv.nn.Activation import BFPActivation
import torch
import torch.nn.functional as F

class CNN10(torch.nn.Module):
    def __init__(self):
        super(CNN10, self).__init__()
        bit = 2

        self.conv1 = DSConv2d(3, 64, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation1 = BFPActivation(bit, 7, 32)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = DSConv2d(64, 64, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation2 = BFPActivation(bit, 7, 32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = DSConv2d(64, 64, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation3 = BFPActivation(bit, 7, 32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = DSConv2d(64, 128, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation4 = BFPActivation(bit, 7, 32)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = DSConv2d(128, 128, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation5 = BFPActivation(bit, 7, 32)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.conv6 = DSConv2d(128, 128, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation6 = BFPActivation(bit, 7, 32)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.conv7 = DSConv2d(128, 256, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation7 = BFPActivation(bit, 7, 32)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.conv8 = DSConv2d(256, 256, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation8 = BFPActivation(bit, 7, 32)
        self.bn8 = torch.nn.BatchNorm2d(256)
        self.conv9 = DSConv2d(256, 512, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation9 = BFPActivation(bit, 7, 32)
        self.bn9 = torch.nn.BatchNorm2d(512)
        self.conv10 = DSConv2d(512, 512, (3, 3), block_size=32, bit=bit, padding=1)
        self.activation10 = BFPActivation(bit, 7, 32)
        self.bn10 = torch.nn.BatchNorm2d(512)

        self.max_pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.max_pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.avg_pool = torch.nn.AvgPool2d(8)

        self.linear = torch.nn.Linear(512, 10)

    def quantize(self):
        self.conv1.quantize()
        self.conv2.quantize()
        self.conv3.quantize()
        self.conv4.quantize()
        self.conv5.quantize()
        self.conv6.quantize()
        self.conv7.quantize()
        self.conv8.quantize()
        self.conv9.quantize()
        self.conv10.quantize()

    def forward(self, x):
        x = F.relu(self.activation1(self.bn1(self.conv1(x))))
        x = F.relu(self.activation2(self.bn2(self.conv2(x))))
        x = F.relu(self.activation3(self.bn3(self.conv3(x))))
        x = self.max_pool1(x)
        x = F.relu(self.activation4(self.bn4(self.conv4(x))))
        x = F.relu(self.activation5(self.bn5(self.conv5(x))))
        x = F.relu(self.activation6(self.bn6(self.conv6(x))))
        x = self.max_pool2(x)
        x = F.relu(self.activation7(self.bn7(self.conv7(x))))
        x = F.relu(self.activation8(self.bn8(self.conv8(x))))
        x = F.relu(self.activation9(self.bn9(self.conv9(x))))
        x = self.activation10(self.bn10(self.conv10(x)))
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

