from .resnet import QUANTIZED_ResNet18, QUANTIZED_ResNet34, QUANTIZED_ResNet50
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["QUANTIZED_ResNet18", "QUANTIZED_ResNet34", "QUANTIZED_ResNet50"]

