'''
(c) Marcelo Genanri 2019
Implementation of torch.autograd.Function that transforms Conv weight into DSConv.

The backward pass is using the Straight-Through Estimator (STE) such that the fp32
weights can be updated using full precision.
'''
# Pytorch
import torch

class DSConvQuant(torch.autograd.Function):
    '''
    Uses least square to find the alpha values as described in DSConv paper.
    Forward method uses a loop over the number of blocks, so not optimal
    when weight size is too deep.
    '''
    @staticmethod
    def __finding_alpha__(original_block, scaled_blck):
        numerator = (original_block*scaled_blck).sum(dim=1)
        denominator = (scaled_blck*scaled_blck).sum(dim=1)
        alpha = numerator/denominator
        final_block = scaled_blck*alpha.unsqueeze(1)
        return final_block, alpha

    @staticmethod
    def __quant_blk__(blcknump, min_v, max_v):
        absblcknump = torch.abs(blcknump)
        _, index_pos = torch.max(absblcknump, dim=1)
        absmax = torch.gather(blcknump, 1, index_pos.unsqueeze(1))

        scaling = min_v/absmax

        scaled_blck = torch.round(scaling*blcknump)
        scaled_blck = torch.clamp(scaled_blck, min=min_v, max=max_v)

        final_block, alpha = DSConvQuant.__finding_alpha__(blcknump, scaled_blck)


        return final_block, scaled_blck, alpha

    @staticmethod
    def forward(ctx, weight, block_size, bit, number_blocks):
        shp = weight.shape
        tensor = weight.data.clone()
        intw = torch.Tensor(shp)
        alpha = torch.Tensor(shp[0], number_blocks, shp[2], shp[3])

        # Use FP32 value aka weight in case bit is None
        if bit is None:
            return tensor, intw, alpha

        blk = block_size
        min_v = -1*pow(2, bit-1)
        max_v = pow(2, bit-1)-1


        for i in range(number_blocks):
            if i == number_blocks-1:
                (tensor[:, i*blk:, ...],
                 intw[:, i*blk:, ...],
                 alpha[:, i, ...]) = \
                 DSConvQuant.__quant_blk__(tensor[:, i*blk:, ...], min_v, max_v)
            else:
                (tensor[:, i*blk:(1+i)*blk, ...],
                 intw[:, i*blk:(1+i)*blk, ...],
                 alpha[:, i, ...]) = \
                 DSConvQuant.__quant_blk__(tensor[:, i*blk:(1+i)*blk, ...], min_v, max_v)

        return tensor, intw, alpha


    @staticmethod
    def backward(ctx, grad_weight, _):
        return grad_weight, None, None, None
