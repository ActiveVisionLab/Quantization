'''
(c) Marcelo Genanri 2019
VGG module adapted from the original pytorch file in order to use DSConv
'''
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from ..DSConv.nn.dsconv2d import DSConv2d
from ..src.bfpactivation import BFPActivation
from ..DSConv.nn.quantized_module import QuantizedModule

MODEL_URLS = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

CFG = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class QuantizedVGG(QuantizedModule):

    def __init__(self, cfg, batch_norm, bits, number_bits, num_classes=1000, init_weights=True):
        super(QuantizedVGG, self).__init__(bits, number_bits)
        self.features = self.make_layers(CFG[cfg], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                bit = self.bits.pop(0) #if i == 0 else self.bits[0]
                conv2d = DSConv2d(in_channels, v, kernel_size=3, block_size=32, bit=bit, padding=1, bias=True)
                bit = self.bits[0]
                activ_layer = [nn.ReLU(inplace=True), BFPActivation(bit, 7, blk=32)] if i != len(cfg)-2 else \
                              [nn.ReLU(inplace=True)]
                activ = nn.Sequential(*activ_layer)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), activ]
                else:
                    layers += [conv2d, activ]
                in_channels = v
        return nn.Sequential(*layers)


class QuantizedVGG11(QuantizedVGG):
    number_bits = 11
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG11, self).__init__('A', False, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg11'], progress=progress)
            self.load_state_dict(state_dict)

class QuantizedVGG11_bn(QuantizedVGG):
    number_bits = 11
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG11_bn, self).__init__('A', True, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg11_bn'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG13(QuantizedVGG):
    number_bits = 13
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG13, self).__init__('B', False, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg13'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG13_bn(QuantizedVGG):
    number_bits = 13
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG13_bn, self).__init__('B', True, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg13_bn'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG16(QuantizedVGG):
    number_bits = 16
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG16, self).__init__('D', False, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg16'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG16_bn(QuantizedVGG):
    number_bits = 16
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG16_bn, self).__init__('D', True, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg16_bn'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG19(QuantizedVGG):
    number_bits = 19
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG19, self).__init__('E', False, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg19'], progress=progress)
            self.load_state_dict(state_dict)


class QuantizedVGG19_bn(QuantizedVGG):
    number_bits = 19
    def __init__(self, bits, block_size, pretrained=False, progress=True, **kwargs):
        super(QuantizedVGG19_bn, self).__init__('E', True, bits, self.number_bits, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg19_bn'], progress=progress)
            self.load_state_dict(state_dict)
