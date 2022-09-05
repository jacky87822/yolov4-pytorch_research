from cProfile import label
from matplotlib import ticker
from pytest import Item
import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
import numpy as np
import struct
from pytorchfi.core import fault_injection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
i="yolov4"
FILE="../../model/"+i+".pt"
model= torch.load(FILE)
model = model.to('cpu')

all=torch.Tensor([]).to('cpu')
all_bn=torch.Tensor([]).to('cpu')
all_conv=torch.Tensor([]).to('cpu')

# find global max exponent
for layer, (name, para) in enumerate(model.named_parameters()): 
    print (name)
    flatten=para.flatten()
    all=torch.cat((all,flatten))
    if "bn" in name:
        all_bn=torch.cat((all_bn,flatten))
    else:
        all_conv=torch.cat((all_conv,flatten))
    
print ("all:",torch.max(all).item(),torch.min(all).item())
print ("all_bn:",torch.max(all_bn).item(),torch.min(all_bn).item())
print ("all_conv:",torch.max(all_conv).item(),torch.min(all_conv).item())

# distplot

fig=plt.figure(figsize=(6,2))
#fig.set_rasterized(True)

fig=sns.histplot(all.detach().numpy(),bins=100,binwidth=0.6, alpha  = 1)
sns.set_style('whitegrid')
plt.yscale("log")
fig.set_axisbelow(True)
plt.grid()
plt.ylabel("# parameters",fontsize=12)
plt.yticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('test3.eps',format='eps',dpi=300)