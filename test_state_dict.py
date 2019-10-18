from models.resnet import QuantizedResNet18
import torch

model = QuantizedResNet18(8, 32, pretrained=True)
model.quantize()
# print(model.state_dict())
# print(model.state_dict_quant())

torch.save(model.state_dict(), 'normal')
torch.save(model.state_dict_quant(), 'quant')
