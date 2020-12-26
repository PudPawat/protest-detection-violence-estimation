import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns # I love this package!
sns.set_style('white')

import torch

# load check point
model_path = 'output/checkpoint.pth.tar'
checkpoint = torch.load('output/checkpoint.pth.tar')
# print("ch",checkpoint)
loss_history_train = checkpoint['loss_history_train']
loss_history_val = checkpoint['loss_history_val']
loss_his_val = [] # change type tensor to numpy for plotting
for loss in loss_history_val:
    loxx = []
    for los in loss:
        loxx.append(los.cpu().numpy())
    loss_his_val.append(loxx)

print(type(loss_history_val[0][0]))
print(loss_history_train[0][0])
loss_train = [np.mean(l) for l in loss_history_train]
loss_val = [np.mean(l) for l in loss_his_val]
plt.plot(loss_train, label = 'Train Loss')
plt.plot(loss_val, label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.legend()
plt.show()