'''
(c) Marcelo Genanri 2019
Parent class of modules that should be quantized.
This does not influence on the torch.nn.Module capabilities (no major overridden methods)
'''
import torch
from DSConv.nn.dsconv2d import DSConv2d


class QuantizedModule(torch.nn.Module):
    '''
    Wrapper to be used on modules to be quantized.
    Just inheriting from this should allow module to be quantized to the bits specified
    '''
    def __init__(self, bits, number_bits):
        super(QuantizedModule, self).__init__()
        if isinstance(bits, int) or (bits is None):
            self.bits = [bits for _ in range(number_bits)]
        else:
            assert number_bits == len(bits)
            self.bits = bits

    @classmethod
    def get_number_layers(cls):
        ''' Returns Number of Layers of this class'''
        return cls.number_bits

    @classmethod
    def get_original_accuracy(cls):
        ''' Returns original top1 and top5 accuracy of this class'''
        return cls.top1, cls.top5

    def forward(self, *inp):
        raise NotImplementedError

    def quantize(self):
        '''
        Calls the quantized function of every module that is a DSConv2d module
        '''
        _ = [m.quantize() for m in self.modules() if isinstance(m, DSConv2d)]

    def state_dict_quant(self, bits=8):
        '''
        Returns dictionary (just like state_dict) of quantized weights.
        For now works only for bits < 8. It is optimized for bits = 4 and bits = 2
        because of uint8 packaging.

        The alpha value of all convolutions (except the first one) is set to float16.
        (This is because when the first one is set to float16 for some reason accuracy
        is severely compromised)
        '''
        _state_dict_quant_ = {}
        first = True
        for name, mod in self.named_modules():
            if isinstance(mod, DSConv2d):
                _state_dict_quant_[name + '.alpha'] = mod.alpha.to(dtype=torch.float) if first \
                                                      else mod.alpha.to(dtype=torch.half)
                first = False
                if bits == 4:
                    _state_dict_quant_[name + '.intw'] = self.__compress_to_4_bits__(mod)
                elif bits == 2:
                    _state_dict_quant_[name + '.intw'] = self.__compress_to_2_bits__(mod)
                else:
                    _state_dict_quant_[name + '.intw'] = mod.intw.to(dtype=torch.int8)

            if isinstance(mod, torch.nn.Linear):
                _state_dict_quant_[name+'.bias'] = mod.bias.to(dtype=torch.half)
                _state_dict_quant_[name+'.weight'] = mod.weight.to(dtype=torch.half)

            if isinstance(mod, torch.nn.BatchNorm2d):
                _state_dict_quant_[name+'.weight'] = mod.weight.to(dtype=torch.half)
                _state_dict_quant_[name+'.bias'] = mod.bias.to(dtype=torch.half)
                _state_dict_quant_[name+'.running_mean'] = mod.running_mean.to(dtype=torch.half)
                _state_dict_quant_[name+'.running_var'] = mod.running_var.to(dtype=torch.half)
                _state_dict_quant_[name+'.num_batches_tracked'] = mod.num_batches_tracked

        return _state_dict_quant_

    def load_state_dict_quant(self, state_dict_quant, block_size=32, bits=8):
        '''
        Used to load the state_dict (just like torch.Tensor.load_state_dict()),
        which was saved using state_dict_quant
        '''
        for name, mod in self.named_modules():
            if isinstance(mod, DSConv2d):
                mod.alpha = state_dict_quant[name + '.alpha'].to(dtype=torch.float)
                mod.alpha = mod.alpha.repeat_interleave(block_size, dim=1)

                if bits == 4:
                    mod.intw = self.__decompress_to_4_bits__(state_dict_quant[name + '.intw'], mod)
                elif bits == 2:
                    mod.intw = self.__decompress_to_2_bits__(state_dict_quant[name + '.intw'], mod)
                else:
                    mod.intw = state_dict_quant[name + '.intw'].to(dtype=torch.float)
                shp = mod.intw.shape
                mod.alpha = mod.alpha[:, :shp[1], ...]  # in case shp[1] is not a multiple of 32

                mod.weight.data = mod.intw*mod.alpha
                mod.quant_w.data = mod.weight.data

            if isinstance(mod, torch.nn.Linear):
                mod.bias.data = state_dict_quant[name+'.bias'].to(dtype=torch.float)
                mod.weight.data = state_dict_quant[name+'.weight'].to(dtype=torch.float)

            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.weight.data = state_dict_quant[name+'.weight'].to(dtype=torch.float)
                mod.bias.data = state_dict_quant[name+'.bias'].to(dtype=torch.float)
                mod.running_mean.data = state_dict_quant[name+'.running_mean'].to(dtype=torch.float)
                mod.running_var.data = state_dict_quant[name+'.running_var'].to(dtype=torch.float)
                mod.num_batches_tracked = state_dict_quant[name+'.num_batches_tracked']

    def __compress_to_2_bits__(self, module):
        ''' Provides the packaging of 2bit weights into uint8 types (25% memory usage)'''
        val = module.intw.flatten()
        val1 = val[:int(val.size()[0]/4)] + 2
        val2 = val[int(val.size()[0]/4):int(val.size()[0]/2)] + 2
        val3 = val[int(val.size()[0]/2):3*int(val.size()[0]/4)] + 2
        val4 = val[3*int(val.size()[0]/4):] + 2

        val_res = val1 + (2**2)*val2 + (2**4)*val3 + (2**6)*val4

        return val_res.to(dtype=torch.uint8)

    def __compress_to_4_bits__(self, module):
        ''' Provides the packaging of 4bit weights into uint8 types (50% memory usage)'''
        val = module.intw.flatten()
        val1 = val[:int(val.size()[0]/2)]+8
        val2 = val[int(val.size()[0]/2):]+8

        val_res = val1 + (2**4)*val2

        return val_res.to(dtype=torch.uint8)

    def __decompress_to_2_bits__(self, tensor, module):
        ''' Provides the unpackaging of 2bit weights into float32 for inference'''
        val_res = tensor
        val4 = (val_res/(2**6)).to(dtype=torch.int32)
        val4 = val4.to(dtype=torch.float32)
        val_res = val_res.to(dtype=torch.float32) - (2**6)*val4

        val3 = (val_res/(2**4)).to(dtype=torch.int32)
        val3 = val3.to(dtype=torch.float32)
        val_res = val_res.to(dtype=torch.float32) - (2**4)*val3

        val2 = (val_res/2**2).to(dtype=torch.int32)
        val2 = val2.to(dtype=torch.float32)

        val1 = val_res.to(dtype=torch.float32) - (2**2)*val2
        val1.to(dtype=torch.float32)

        val1 = val1 - 2
        val2 = val2 - 2
        val3 = val3 - 2
        val4 = val4 - 2

        val = torch.cat((val1, val2, val3, val4))
        val = val.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        val = val.reshape(module.intw.shape)

        return val

    def __decompress_to_4_bits__(self, tensor, module):
        ''' Provides the unpackaging of 4bit weights into float32 for inference'''
        val_res = tensor
        val2 = (val_res/(2**4)).to(dtype=torch.int32)
        val2 = val2.to(dtype=torch.float32)
        val1 = val_res.to(dtype=torch.float32) - (2**4)*val2
        val1 = val1.to(dtype=torch.float32)

        val1 = val1 - 8
        val2 = val2 - 8

        val = torch.cat((val1, val2))
        val = val.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        val = val.reshape(module.intw.shape)

        return val
