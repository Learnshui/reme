from model.S_V2 import S_V2_Net
import torch
x=torch.rand([1,3,224,224])
model = S_V2_Net()
y=model(x)
print(y.shape)
print('net total parameters:', sum(param.numel() for param in model.parameters()))

# torch.Size([1, 1, 224, 224])
# net total parameters: 34870232
# torch.Size([1, 1, 224, 224])
# net total parameters: 34890683