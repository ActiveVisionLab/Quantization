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
        # destination = super(QuantizedModule, self).state_dict(destination, prefix, keep_vars)
        # print(destination.keys())
        # input('')
        _state_dict_quant_ = {}
        for name, m in self.named_modules():
            if isinstance(m, DSConv2d):
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

            
    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     print("I am overwriting state_dict")
    #     destination = super(QuantizedModule, self).state_dict(destination, prefix, keep_vars)
    #     print(destination.keys())
    #     input('')
    #     conv_dests = [k for k in destination.keys() if isinstance(getattr(self, ''.join(k.split('.')[:-1])), DSConv2d)]
    #     print(conv_dests)
    #     input('')

    #     for k in destination.keys():
    #         print('.'.join(k.split('.')[:-1]))
    #         # print(getattr(self, k[:5]))
    #         input('')
        

    #     return destination
