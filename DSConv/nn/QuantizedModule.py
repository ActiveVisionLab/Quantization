import torch
from DSConv.nn.DSConv2d import DSConv2d


class QuantizedModule(torch.nn.Module):
    def __init__(self, bits, number_bits):
        super(QuantizedModule, self).__init__()
        if type(bits) == int or bits == None:
            self.bits = [bits for _ in range(number_bits)]
        else:
            assert(number_bits == len(bits))
            self.bits = bits

    def quantize(self):
        [m.quantize() for m in self.modules() if isinstance(m, DSConv2d)]

    def state_dict_quant(self, destination = None, prefix='', keep_vars=False, block_size=32, bits=8):
        _state_dict_quant_ = {}
        count = 0
        for name, m in self.named_modules():
            if isinstance(m, DSConv2d):
                _state_dict_quant_[name + '.alpha'] = m.alpha.to(dtype=torch.float32) if count == 0 else m.alpha.to(dtype=torch.float16)
                count += 1
                if bits==4:
                    _state_dict_quant_[name + '.intw'] = self.__compress_to_4_bits__(m)
                elif bits==2:
                    _state_dict_quant_[name + '.intw'] = self.__compress_to_2_bits__(m)
                else: 
                    _state_dict_quant_[name + '.intw'] = m.intw.to(dtype=torch.int8)
            
            if isinstance(m, torch.nn.Linear):
                _state_dict_quant_[name+'.bias'] = m.bias.to(dtype=torch.float16)
                _state_dict_quant_[name+'.weight'] = m.weight.to(dtype=torch.float16)

            if isinstance(m, torch.nn.BatchNorm2d):
                _state_dict_quant_[name+'.weight'] = m.weight.to(dtype=torch.float16)
                _state_dict_quant_[name+'.bias'] = m.bias.to(dtype=torch.float16)
                _state_dict_quant_[name+'.running_mean'] = m.running_mean.to(dtype=torch.float16)
                _state_dict_quant_[name+'.running_var'] = m.running_var.to(dtype=torch.float16)
                _state_dict_quant_[name+'.num_batches_tracked'] = m.num_batches_tracked

        return _state_dict_quant_

    def load_state_dict_quant(self, state_dict_quant, block_size=32, bits=8):
        for name, m in self.named_modules():
            if isinstance(m, DSConv2d):
                m.alpha = state_dict_quant[name + '.alpha'].to(dtype=torch.float32)
                m.alpha = m.alpha.repeat_interleave(block_size, dim=1)

                if bits == 4:
                    m.intw = self.__decompress_to_4_bits__(state_dict_quant[name + '.intw'], m)
                elif bits == 2:
                    m.intw = self.__decompress_to_2_bits__(state_dict_quant[name + '.intw'], m)
                else:
                    m.intw = state_dict_quant[name + '.intw'].to(dtype=torch.float32)
                shp = m.intw.shape
                m.alpha = m.alpha[:, :shp[1], ...]  # in case shp[1] is not a multiple of 32

                m.weight.data = m.intw*m.alpha
                m.quant_w.data = m.weight.data

            if isinstance(m, torch.nn.Linear):
                m.bias.data = state_dict_quant[name+'.bias'].to(dtype=torch.float32)
                m.weight.data = state_dict_quant[name+'.weight'].to(dtype=torch.float32)

            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data = state_dict_quant[name+'.weight'].to(dtype=torch.float32)
                m.bias.data = state_dict_quant[name+'.bias'].to(dtype=torch.float32)
                m.running_mean.data = state_dict_quant[name+'.running_mean'].to(dtype=torch.float32)
                m.running_var.data = state_dict_quant[name+'.running_var'].to(dtype=torch.float32)
                m.num_batches_tracked = state_dict_quant[name+'.num_batches_tracked']

    def __compress_to_2_bits__(self, module):
        val = module.intw.flatten()
        val1 = val[:int(val.size()[0]/4)] + 2
        val2 = val[int(val.size()[0]/4):int(val.size()[0]/2)] + 2
        val3 = val[int(val.size()[0]/2):3*int(val.size()[0]/4)] + 2
        val4 = val[3*int(val.size()[0]/4):] + 2

        val_res = val1 + (2**2)*val2 + (2**4)*val3 + (2**6)*val4

        return val_res.to(dtype=torch.uint8)

    def __decompress_to_2_bits__(self, tensor, module):
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

    def __compress_to_4_bits__(self, module):
        val = module.intw.flatten()
        val1 = val[:int(val.size()[0]/2)]+8
        val2 = val[int(val.size()[0]/2):]+8
        
        val_res = val1 + (2**4)*val2

        return val_res.to(dtype=torch.uint8)
    
    def __decompress_to_4_bits__(self, tensor, module):
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