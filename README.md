# Block Floating Point (BFP) and [DSConv](https://arxiv.org/abs/1901.01928) with GPU support

This repo implements BFP and DSConv in cuda kernels to be used with [PyTorch](https://pytorch.org/)

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* CUB == 1.8

## Build

1. Download [CUB](https://github.com/NVlabs/cub) and put it in `/home/your_username/libs/` (or the file indicated at `NUQ/BlackBox/Quantization/src/setup.py:22`)
2. `cd /path/to/NUQ/BlackBox/Quantization/src/` then `python build_ext --inplace`.
