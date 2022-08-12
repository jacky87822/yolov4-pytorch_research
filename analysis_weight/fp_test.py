from cProfile import label
from matplotlib import ticker
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


def ieee_754_conversion(n, sgn_len=1, exp_len=8, mant_len=23):
    """
    Converts an arbitrary precision Floating Point number.
    Note: Since the calculations made by python inherently use floats, the accuracy is poor at high precision.
    :param n: An unsigned integer of length sgn_len + exp_len + mant_len to be decoded as a float
    :param sgn_len: number of sign bits
    :param exp_len: number of exponent bits
    :param mant_len: number of mantissa bits
    :return: IEEE 754 Floating Point representation of the number n
    """
    if n >= 2 ** (sgn_len + exp_len + mant_len):
        raise ValueError("Number n is longer than prescribed parameters allows")

    sign = (n & (2 ** sgn_len - 1) * (2 ** (exp_len + mant_len))) >> (exp_len + mant_len)
    exponent_raw = (n & ((2 ** exp_len - 1) * (2 ** mant_len))) >> mant_len
    mantissa = n & (2 ** mant_len - 1)

    sign_mult = 1
    if sign == 1:
        sign_mult = -1

    if exponent_raw == 2 ** exp_len - 1:  # Could be Inf or NaN
        if mantissa == 2 ** mant_len - 1:
            return float('nan')  # NaN

        return sign_mult * float('inf')  # Inf

    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)

    if exponent_raw == 0:
        mant_mult = 0  # Gradual Underflow
    else:
        mant_mult = 1

    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
def binary_h(num):
    return bin(np.float16(num.cpu().detach().numpy()).view('H'))[2:].zfill(16)

def show_bin(i,format):
    if format=="fp32":
        bin_32i=binary(i)
        bin_32i_list=[*bin_32i]
        
        sign = ''.join(bin_32i_list[0])
        expn = ''.join(bin_32i_list[1:9])
        mant = ''.join(bin_32i_list[9:])
    elif format=="fp16":
        bin_16i=binary_h(i)
        bin_16i_list=[*bin_16i]
        
        sign= ''.join(bin_16i_list[0])
        expn = ''.join(bin_16i_list[1:6])
        mant = ''.join(bin_16i_list[6:])
    elif format=="bf16":
        bin_32i=binary(i)
        bin_32i_list=[*bin_32i]
        
        sign = ''.join(bin_32i_list[0])
        expn = ''.join(bin_32i_list[1:9])
        mant = ''.join(bin_32i_list[9:16])
    return sign,expn,mant
    
i = 'yolov4'
FILE="../../model/"+i+".pt"

model= torch.load(FILE)
format="fp16"
model = model.to('cuda')

for layer, (name, para) in enumerate(model.named_parameters()): 
    if layer==1:
        for i in para:
            print ("=========================================================")
            print ("fp32:",i.item())
            print (show_bin(i,"fp32"))
            print (show_bin(i,"fp16"))
            sign,expn,mant=show_bin(i,"fp32")
            print (ieee_754_conversion(int((sign+expn+mant),2), sgn_len=1, exp_len=8, mant_len=23))
            print ('--------------------------------')
            
            i_fp16=i.half()
            print ("fp16:",i_fp16.item())
            print (show_bin(i_fp16,"fp32"))
            print (show_bin(i_fp16,"fp16"))
            sign,expn,mant=show_bin(i_fp16,"fp16")
            print (ieee_754_conversion(int((sign+expn+mant),2), sgn_len=1, exp_len=5, mant_len=10))
            print ('--------------------------------')
            
            i_bp16=i.to(torch.bfloat16)
            print ("bp16:",i_bp16.item())
            print (show_bin(i_bp16,"bf16"))
            sign,expn,mant=show_bin(i_fp16,"bf16")
            print (ieee_754_conversion(int((sign+expn+mant),2), sgn_len=1, exp_len=8, mant_len=7))
            
            
            
