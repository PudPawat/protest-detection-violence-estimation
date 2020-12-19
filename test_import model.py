import torch
from resnext_wsl import resnext101_32x48d_wsl

model=resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 
print(model)

# pretrained_dict=torch.load('ig_resnext101_32x48-3e41cc8a.pth',map_location='cpu')['model']

# model_dict = model.state_dict()
# for k in model_dict.keys():
#     if(('module.'+k) in pretrained_dict.keys()):
#         model_dict[k]=pretrained_dict.get(('module.'+k))
# model.load_state_dict(model_dict)

# print(model)