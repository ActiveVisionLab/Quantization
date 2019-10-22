from models.resnet import QuantizedResNet18
from test_models import test
import torch

## Creating quant4 and quant2
bits = 4
model = QuantizedResNet18(bits, 32, pretrained=True)
model.quantize()
torch.save(model.state_dict_quant(bits=bits), 'quant4.pth')
# correct1, correct5, total = test(model, 128)
# print(f"Results from unloaded quant4 model: Top1 {correct1/total}")

bits = 2
model = QuantizedResNet18(bits, 32, pretrained=True)
model.quantize()
torch.save(model.state_dict_quant(bits=bits), 'quant2.pth')
# correct1, correct5, total = test(model, 128)
# print(f"Results from unloaded quant2 model: Top1 {correct1/total}")

## Testing quant4 and quant2
bits=4
model = QuantizedResNet18(bits, 32, pretrained=False)
model.load_state_dict_quant(torch.load('quant4.pth'), bits=bits)
correct1, correct5, total = test(model, 128)
print(f"Results from loaded quant4 model: Top1 {correct1/total}")

bits=2
model = QuantizedResNet18(bits, 32, pretrained=False)
model.load_state_dict_quant(torch.load('quant2.pth'), bits=bits)
correct1, correct5, total = test(model, 128)
print(f"Results from loaded quant2 model: Top1 {correct1/total}")
