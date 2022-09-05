import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
import numpy as np
import struct
import cupy
import time
from colorama import Fore, Back, Style
import logging
def mantissa_fp2bin(x, bit=23):
    x=torch.abs(x)
    x=torch.multiply(x,2**(bit+1))
    y=dec2bin(x.int(), bits=bit)
    return y

def dec2bin(x, bits=8):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def inject(para_bin,inject_map,operation,device):
    if operation=="sa0":
        para_bin_zero = torch.zeros(para_bin.shape).to(device)
        new_para_bin=torch.where(inject_map > 0, para_bin_zero, para_bin.to(device))
    elif operation=="sa1":
        para_bin_ones = torch.ones(para_bin.shape).to(device)
        new_para_bin=torch.where(inject_map > 0, para_bin_ones, para_bin.to(device))
    elif operation=="bf":
        new_para_bin=torch.bitwise_xor(para_bin.int().to(device),inject_map.int().to(device))
    else:
        new_para_bin=para_bin
    return new_para_bin.to(device)
    
def bin2fp(para_bin,device,exp_bit=8,mant_bit=23):
    # fp16 -126 should be modified
    new_sign=para_bin[...,0]*-2+1
    new_exp=bin2dec(para_bin[...,1:exp_bit+1], bits=exp_bit)-126
    MSBs=torch.ones(para_bin[...,0].shape).to(device)
    LSBs=torch.zeros(para_bin[...,0].shape).to(device)
    MSBs.unsqueeze_(-1)
    LSBs.unsqueeze_(-1)
    
    # accurate ones
    #new_mant=torch.cat((para_bin[...,exp_bit+1:],LSBs,LSBs,LSBs),-1)
    #new_mant_1=bin2dec(new_mant, bits=mant_bit+3)*(2**(-mant_bit-3))
    #new_mant=torch.cat((MSBs,para_bin[...,exp_bit+1:],LSBs,LSBs,LSBs),-1)
    #new_mant_2=bin2dec(new_mant, bits=mant_bit+1+3).int()*(2**(-mant_bit-1-3))
    #new_mant=torch.where(new_exp==-126,new_mant_1,new_mant_2)
    
    new_mant=torch.cat((MSBs,para_bin[...,exp_bit+1:],LSBs,LSBs,LSBs),-1)
    new_mant=bin2dec(new_mant, bits=mant_bit+4)*(2**(-mant_bit-4))
    
    new_mant=torch.multiply(new_mant,new_sign)
    new_para=torch.ldexp(new_mant, new_exp)
    #print (new_exp)
    return new_para



def inj_model(model_name,device):
    model = torch.load(model_name)
    model=model.to(device)
    start = time.time()    
                
    with torch.no_grad():
        all_sum=torch.zeros(32).to(device)
        para_num=0
        for layer, (name, para) in enumerate(model.named_parameters()): 
            print (name)
            #print (binary(para))
            #print ("--------------------------------")
            para=para.flatten()
            para_num+=len(para)
            mantissa, exponent = torch.frexp(para)
            sign=torch.abs((para.sign()-1)/2)
            exp_bin=dec2bin(exponent+126, bits=8)
            mant_bin=mantissa_fp2bin(mantissa, bit=23)
            
            sign.unsqueeze_(-1)
            para_bin=torch.cat((sign,exp_bin,mant_bin),-1)
            sum=torch.cumsum(para_bin, dim=0)[-1]
            all_sum=sum+all_sum
        print (all_sum)   
        print (all_sum/para_num)
        
        
        
inj_model("../model/yolov4.pt","cuda:1")
