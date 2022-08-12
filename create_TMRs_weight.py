import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
name_list=['yolov4']
import numpy as np
import struct

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



for i in name_list:
    
    FILE="../model/"+i+".pt"
    FILE_TMR="../model/TMRs_"+i+".pt"
    
    model_free = torch.load(FILE)
    #model = torch.load(FILE_TMR)
    #model = model.to('cuda')
    model_free = model_free.to('cuda')
    with torch.no_grad():
        for layer, (name, para) in enumerate(model_free.named_parameters()): 
            print (layer,name)
            para_shape=para.shape

            if len(para_shape) == 4: 
                for k in range (para_shape[0]):
                    for c in range (para_shape[1]):
                        for h in range (para_shape[2]):
                            for w in range (para_shape[3]):
                                #print ('value',model.state_dict()[name][k,c,h,w])
                                value=str(binary(model_free.state_dict()[name][k,c,h,w]))
                                value_split=[*value]
                                #print ('value',value)
                                new_value_list=value_split[:16]+value_split[0:8]+value_split[0:8]
                                new_value = ''.join(new_value_list)
                                #print ('new_va',new_value)
                                model_free.state_dict()[name][k,c,h,w]=ieee_754_conversion(int(new_value,2))
                                #print ('new_va',model.state_dict()[name][k,c,h,w])
            elif len(para_shape) == 1:
                print ("//---------")
                for i in range(para_shape[0]):
                    value=str(binary(model_free.state_dict()[name][i]))
                    value_split=[*value]
                    new_value_list=value_split[:16]+value_split[0:8]+value_split[0:8]
                    new_value = ''.join(new_value_list)
                    model_free.state_dict()[name][i]=ieee_754_conversion(int(new_value,2))
            else:
                print (para_shape)
                exit()


    #FILE_ERR="../model/"+i+"_ERR.pt"
    torch.save(model_free, FILE_TMR)
    
    print ('//------------------------------')
    model = torch.load(FILE_TMR)
    model = model.to('cuda')

    for layer, (name, para) in enumerate(model.named_parameters()): 
        print ("===================================")
        if layer ==2:
            break
        else:
            para_shape=para.shape
            if len(para_shape) == 4:
                for k in range (para_shape[0]):
                    for c in range (para_shape[1]):
                        for h in range (para_shape[2]):
                            for w in range (para_shape[3]):
                                print (binary(model.state_dict()[name][k,c,h,w]))
            else:
                for i in range(para_shape[0]):
                    print (binary(model.state_dict()[name][i]))
