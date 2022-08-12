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

def draw_fig (value,i,layer_name=None):
    if layer_name:
        os.system(r"rm -rf ./layers/"+layer_name+'.jpg')
    else:
        os.system(r"rm -rf ./layers/"+i+'.jpg')
    #os.system(r"rm -rf ./layers/*")
    
    sorted_all, indices = torch.sort(value)
    mantissa, exponent = torch.frexp(sorted_all)
    x,y=exponent.unique(return_counts=True)
    y_count = torch.div(y,(torch.cumsum(y, dim=0)[-1])*0.01)
    y_count = torch.cumsum(y_count,dim=0)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_count= y_count.cpu().numpy()
    if len(x) > 16:
        x_label=range(0,len(x),2) #x[::2]
    else:
        x_label=range(0,len(x),1)
    print (x_label)

    fig = plt.figure(figsize=(12,9),dpi=300)
    ax1 = fig.add_subplot(111)
    ax1 = plt.yscale("log")
    ax1 = sns.barplot(x=x,y=y,palette='Blues_d', label='Barplot label')
    
    ax1.set_ylabel("Count", fontsize = 20, color='blue')
    ax1.set_xlabel("Exponent", fontsize = 20)
    if layer_name:
        ax1.set_title("Distribution of "+i+" weights in "+layer_name, fontsize = 24)
    else:
        plt.title("Distribution of "+i+" weights", fontsize = 24)
        
    ax1.tick_params(axis="both",labelsize=16)
    
    
    ax2 = ax1.twinx() 
    
    sns.pointplot(x=x,y=y_count,scale = 1, ax=ax2, color="red", label='pointplot')
    ax2.set_ylabel("Cumulative percentage", fontsize = 20, color='red')
    ax2.tick_params(axis="both",labelsize=16)
    ax1.set_xticks(ticks=x_label)
    
    #plt.legend(loc='upper left', fontsize = 14) 
    if layer_name:
        plt.savefig('./layers/'+layer_name+'.jpg')
    else:
        plt.savefig('./layers/'+i+'.jpg')
    plt.show()
    
    
i = 'yolov4'
aa=[8,16,32]
mask_count_list=[]
global_base=True
save=True

for a in aa:
    FILE="../../model/"+i+".pt"
    NEW_FILE="../../model/"+i+"_trunc_"+str(a)+".pt"

    os.system(r"rm -rf "+NEW_FILE)

    model= torch.load(FILE)
    model = model.to('cuda')
    
    all=torch.Tensor([]).to('cuda')
    
    
    # find global max exponent
    for layer, (name, para) in enumerate(model.named_parameters()): 
        flatten=para.flatten()
        all=torch.cat((all,flatten))
    all_mantissa, all_exponent = torch.frexp(all)
    global_exp=torch.max(all_exponent)
    print ("gobal exp value :",global_exp)
    if global_base:
        NEW_FILE=NEW_FILE.split('.pt')[0]+'_global.pt'
    
    #os.system(r"rm -rf ./layers/*")
    all=torch.Tensor([]).to('cuda')
    all_conv=torch.Tensor([]).to('cuda')
    all_bn=torch.Tensor([]).to('cuda')

    trunc=torch.Tensor([]).to('cuda')
    trunc_conv=torch.Tensor([]).to('cuda')
    trunc_bn=torch.Tensor([]).to('cuda')
    
    with torch.no_grad():
        mask_count=0
        for layer, (name, para) in enumerate(model.named_parameters()): 
            if "bias" not in name:
                print ("============================================")
                print (name)
                
                flatten=para.flatten()
                #draw_fig (flatten,i,layer_name=name.split('.')[2])
                sorted_flatten, indices = torch.sort(flatten)
                mantissa, exponent = torch.frexp(sorted_flatten)
                print ("original:",torch.max(exponent),torch.min(exponent))
                ############################################################################
                
                all=torch.cat((all,flatten))
                if "bn" in name:
                    all_bn=torch.cat((all_bn,flatten))
                elif "conv" in name:
                    all_conv=torch.cat((all_conv,flatten))
                
                ############################################################################
                
                para_mantissa, para_exponent = torch.frexp(para)
                if global_base:
                    new_para=torch.where(para_exponent > (global_exp-a), para, torch.tensor(0.).to("cuda"))
                else:
                    new_para=torch.where(para_exponent > (torch.max(exponent)-a), para, torch.tensor(0.).to("cuda"))
                
                
                para_mantissa, para_exponent = torch.frexp(new_para)
                print ("Approximate:",torch.max(para_exponent),torch.min(para_exponent))
                #draw_fig (new_para.flatten(),i,layer_name=name.split('.')[2]+"_trunc"+str(a))
                
                ############################################################################
                
                error_wo=torch.sub(para,new_para)
                error_wo=torch.where(error_wo==0, torch.tensor(0.).to("cuda"), torch.tensor(1.).to("cuda"))
                error_wo, count = torch.unique(error_wo,return_counts=True, sorted=True)
                if len(error_wo)>1:
                    print ("============================================")
                    print (name)
                    print ("error_wo: ",error_wo,count[1])
                    mask_count=mask_count+count[1]
                
                ############################################################################
                trunc=torch.cat((trunc,new_para.flatten()))
                if "bn" in name:
                    trunc_bn=torch.cat((trunc_bn,new_para.flatten()))
                elif "conv" in name:
                    trunc_conv=torch.cat((trunc_conv,new_para.flatten()))
        
        mask_count_list.append(mask_count)

    '''
    draw_fig (all,i)
    draw_fig (all_conv,i+"_conv")
    draw_fig (all_bn,i+"_bn")

    draw_fig (trunc,i+"_trunc"+str(a))
    draw_fig (trunc_conv,i+"_conv_trunc"+str(a))
    draw_fig (trunc_bn,i+"_bn_trunc"+str(a))
    '''
    
    model= torch.load(FILE)
    model = model.to('cuda')
    for layer, (name, para) in enumerate(model.named_parameters()): 
        flatten=para.flatten()
        mantissa, exponent = torch.frexp(sorted_flatten)
        para_mantissa, para_exponent = torch.frexp(para)
        if global_base:
            new_para=torch.where(para_exponent > (global_exp-a), para, torch.tensor(0.).to("cuda"))
        else:
            new_para=torch.where(para_exponent > (torch.max(exponent)-a), para, torch.tensor(0.).to("cuda"))
        model.state_dict()[name].data.copy_(new_para)
    if save: 
        torch.save(model, NEW_FILE)
    
    

print ("\n\n============================================")
for c,i in enumerate(mask_count_list):
    rate= float (100*i/float(len(all)))
    print ("truncate into",aa[c],": ",rate,"%")  
    