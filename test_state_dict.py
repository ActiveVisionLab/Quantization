from models.resnet import QuantizedResNet18
from test_models import test
import torch

model = QuantizedResNet18(4, 32, pretrained=True)
model.quantize()

print(test(model, 128))

m1 = model.conv1.quant_w

# print(model.state_dict())
# print(model.state_dict_quant())

torch.save(model.state_dict(), 'normal')
torch.save(model.state_dict_quant(), 'quant')

model = QuantizedResNet18(4, 32, pretrained=False)
model.load_state_dict_quant(torch.load('quant'))
m2 = model.conv1.weight
print(test(model, 128))
# input('')
# print(m2)
# input('')
# print(m1-m2)

# print(torch.load('quant'))
