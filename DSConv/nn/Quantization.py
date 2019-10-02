# Other
import math

# Pytorch
import torch
import torch.nn as nn


class DSConvQuant(torch.autograd.Function):

    @staticmethod
    def __finding_alpha__(original_block, scaled_blck):
        numerator = (original_block*scaled_blck).sum(dim=1)
        denominator = (scaled_blck*scaled_blck).sum(dim=1)
        alpha = numerator/denominator
        final_block = scaled_blck*alpha.unsqueeze(1)
        return final_block, alpha

    @staticmethod
    def quant_blk(blcknump, minV, maxV):
        absblcknump = torch.abs(blcknump)
        _, indexPos = torch.max(absblcknump, dim=1)
        absmax = torch.gather(blcknump, 1, indexPos.unsqueeze(1))

        scaling = minV/absmax

        scaled_blck = torch.round(scaling*blcknump)
        scaled_blck = torch.clamp(scaled_blck, min=minV, max=maxV)

        final_block, alpha = DSConvQuant.__finding_alpha__(blcknump, scaled_blck)


        return final_block, scaled_blck, alpha

    @staticmethod
    def forward(ctx, weight, block_size, bit, number_blocks):
        blk = block_size
        minV = -1*pow(2, bit-1)
        maxV = pow(2, bit-1)-1

        shp = weight.shape
        tensor = weight.data.clone()
        intw = torch.Tensor(shp)
        alpha = torch.Tensor(shp[0], number_blocks, shp[2], shp[3])

        for i in range(number_blocks):
            if i == number_blocks-1:
                (tensor[:, i*blk:, ...],
                intw[:, i*blk:, ...],
                alpha[:, i, ...]) = DSConvQuant.quant_blk(tensor[:, i*blk:, ...], minV, maxV)
            else:
                (tensor[:, i*blk:(1+i)*blk, ...],
                intw[:, i*blk:(1+i)*blk, ...],
                alpha[:, i, ...]) = DSConvQuant.quant_blk(tensor[:, i*blk:(1+i)*blk, ...], minV, maxV)

        return tensor, intw, alpha


    @staticmethod
    def backward(ctx, grad_weight, grad_int, grad_alpha):
        return grad_weight, None, None, None

