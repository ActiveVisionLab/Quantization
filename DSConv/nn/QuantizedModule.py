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

    def state_dict_quant(self, destination = None, prefix='', keep_vars=False):
        _state_dict_quant_ = {}
        for name, m in self.named_modules():
            if isinstance(m, DSConv2d):
                print(name)
                _state_dict_quant_[name + '.alpha'] = m.alpha
                _state_dict_quant_[name + '.intw'] = m.intw.to(dtype=torch.int8)
            
            if isinstance(m, torch.nn.Linear):
                _state_dict_quant_[name+'.bias'] = m.bias
                _state_dict_quant_[name+'.weight'] = m.weight

            if isinstance(m, torch.nn.BatchNorm2d):
                _state_dict_quant_[name+'.weight'] = m.weight
                _state_dict_quant_[name+'.bias'] = m.bias
                _state_dict_quant_[name+'.running_mean'] = m.running_mean
                _state_dict_quant_[name+'.running_var']  = m.running_var
                _state_dict_quant_[name+'.num_batches_tracked'] = m.num_batches_tracked

        return _state_dict_quant_

    def load_state_dict_quant(self, state_dict_quant):
        for name, m in self.named_modules():
            if isinstance(m, DSConv2d):
                m.alpha = state_dict_quant[name + '.alpha']
                m.alpha = m.alpha.repeat_interleave(32, dim = 1)
                m.intw = state_dict_quant[name + '.intw'].to(dtype=torch.float32)
                shp = m.intw.shape
                m.alpha = m.alpha[:, :shp[1], ...]  # in case shp[1] is not a multiple of 32 

                m.weight.data = m.intw*m.alpha
                m.quant_w.data = m.weight.data

            if isinstance(m, torch.nn.Linear):
                m.bias = state_dict_quant[name+'.bias']
                m.weight = state_dict_quant[name+'.weight']

            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight = state_dict_quant[name+'.weight']
                m.bias = state_dict_quant[name+'.bias']
                m.running_mean = state_dict_quant[name+'.running_mean']
                m.running_var = state_dict_quant[name+'.running_var']
                m.num_batches_tracked = state_dict_quant[name+'.num_batches_tracked']

