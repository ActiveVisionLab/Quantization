"""
(c) Marcelo Genanri 2019
ResNet module adapted from the original pytorch file in order to use DSConv
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from ..DSConv.nn.dsconv2d import DSConv2d
from ..src.bfpactivation import BFPActivation
from ..DSConv.nn.quantized_module import QuantizedModule

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, block_size, bit, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return DSConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        block_size=block_size,
        bit=bit,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, block_size, bit, stride=1):
    """1x1 convolution"""
    return DSConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        block_size=block_size,
        bit=bit,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        bits,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        block_size=32,
        final=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        bit = bits.pop(0)
        downsample_bit = bit
        self.conv1 = conv3x3(inplanes, planes, block_size, bit, stride)

        bit = bits.pop(0)
        self.activation1 = BFPActivation(bit, block_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, block_size, bit)

        if final:
            self.seq = self.relu
        else:
            bit = bits[0]
            self.activation2 = BFPActivation(bit, block_size)
            self.seq = nn.Sequential(self.relu, self.activation2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        if downsample is not None:
            self.downsample[0].update_bit(downsample_bit)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.seq(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        bits,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        block_size=32,
        final=False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        bit = bits.pop(0)
        downsample_bit = bit
        self.conv1 = conv1x1(inplanes, width, block_size, bit)

        bit = bits.pop(0)
        self.activation1 = BFPActivation(bit, block_size)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, block_size, bit, stride, groups, dilation)

        bit = bits.pop(0)
        self.activation2 = BFPActivation(bit, block_size)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, block_size, bit)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if final:
            self.seq = self.relu
        else:
            bit = bits[0]
            self.activation3 = BFPActivation(bit, block_size)
            self.seq = nn.Sequential(self.relu, self.activation2)

        self.downsample = downsample
        if downsample is not None:
            self.downsample[0].update_bit(downsample_bit)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.seq(out)

        return out


class ResNet(QuantizedModule):
    def __init__(
        self,
        block,
        layers,
        bits,
        number_bits,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__(bits, number_bits)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        bit = self.bits.pop(0)
        self.conv1 = DSConv2d(
            3,
            self.inplanes,
            kernel_size=7,
            block_size=32,
            bit=bit,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        bit = self.bits[0]
        self.activation1 = BFPActivation(bit, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            final=True,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for mod in self.modules():
            if isinstance(mod, DSConv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for mod in self.modules():
                if isinstance(mod, Bottleneck):
                    nn.init.constant_(mod.bn3.weight, 0)
                elif isinstance(mod, BasicBlock):
                    nn.init.constant_(mod.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, block_size=32, final=False
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion, block_size, None, stride
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                self.bits,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            __final__ = False
            if final and i == blocks - 1:
                __final__ = True
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    self.bits,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    final=__final__,
                )
            )

        return nn.Sequential(*layers)

    def update_bits(self, bits):
        for name, mod in self.named_modules():
            if isinstance(mod, DSConv2d):
                bit = bits.pop(0) if "downsample" not in name else bit
                if "conv1" in name:
                    downsample_bit = bit
                if "downsample" in name:
                    mod.update_bit(downsample_bit)
                else:
                    mod.update_bit(bit)
            if isinstance(mod, BFPActivation):
                bit = bits[0]
                mod.update_mantissa(bit)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.activation1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class QuantizedResNet18(ResNet):
    number_bits = 17
    top1 = 69.76
    top5 = 89.08

    def __init__(self, bits, block_size, pretrained=False, progress=False, **kwargs):
        super(QuantizedResNet18, self).__init__(
            BasicBlock, [2, 2, 2, 2], bits, self.number_bits, **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet18"], progress=progress
            )
            self.load_state_dict(state_dict)


class QuantizedResNet34(ResNet):
    number_bits = 33
    top1 = 73.3
    top5 = 91.42

    def __init__(self, bits, block_size, pretrained=False, progress=False, **kwargs):
        super(QuantizedResNet34, self).__init__(
            BasicBlock, [3, 4, 6, 3], bits, self.number_bits, **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet34"], progress=progress
            )
            self.load_state_dict(state_dict)


class QuantizedResNet50(ResNet):
    number_bits = 49
    top1 = 76.15
    top5 = 92.87

    def __init__(self, bits, block_size, pretrained=False, progress=False, **kwargs):
        super(QuantizedResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], bits, self.number_bits, **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet50"], progress=progress
            )
            self.load_state_dict(state_dict)


class QuantizedResNet101(ResNet):
    number_bits = 104

    def __init__(self, bits, block_size, pretrained=False, progress=False, **kwargs):
        super(QuantizedResNet101, self).__init__(
            Bottleneck, [3, 4, 23, 3], bits, self.number_bits, **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet101"], progress=progress
            )
            self.load_state_dict(state_dict)


class QuantizedResNet152(ResNet):
    number_bits = 154

    def __init__(self, bits, block_size, pretrained=False, progress=False, **kwargs):
        super(QuantizedResNet152, self).__init__(
            Bottleneck, [3, 8, 36, 3], bits, self.number_bits, **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet152"], progress=progress
            )
            self.load_state_dict(state_dict)
