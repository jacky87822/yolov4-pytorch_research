import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
import numpy as np
import struct
import cupy
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
def binary_h(num):
    return bin(np.float16(num.cpu().detach().numpy()).view('H'))[2:].zfill(16)
def int_to_bits(x, bits=None, dtype=torch.uint8):
    assert not(x.is_floating_point() or x.is_complex()), "x isn't an integer type"
    if bits is None: bits = x.element_size() * 8
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)

def dec2bin(x, bits=8):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

blind_list=[4]

for blind in blind_list:
    test=False
    i = 'yolov4'
    FILE="../model/"+i+".pt"
    FILE_half="../model/"+i+"_SECDED.pt"
    format="fp32"
    if format=="fp32":
        FILE_half="../model/"+i+"_SECDED.pt"


    if format=="fp16":
        index=12
    if format=="bf16":
        index=9
    elif format=="fp32":
        index=25
    def find(tensor, values):
        return torch.nonzero(tensor[..., None] == values)
    count=0
    os.system(r"rm "+FILE_half)
    os.system(r"cp "+FILE+" "+FILE_half)
    model_free = torch.load(FILE)
    model_free = model_free.to('cuda')


    with torch.no_grad():
        
        for layer, (name, para) in enumerate(model_free.named_parameters()): 
            print (name)
            if format=="fp16":
                para=para.half()
            if format=="bf16":
                para=para.bfloat16()
                
            mantissa, exponent = torch.frexp(para)
            bin=dec2bin(exponent+126, bits=8)
            #print ("para:",para.shape)
            #print ("bin shape:",bin.shape)
            #print ("bin:",bin[..., 0].shape)
            #print (bin[..., 0]+bin[..., 1])
            #print (abs((para.sign()-1)/2))
            sign=abs((para.sign()-1)/2)
            c1=(sign+bin[...,0]+bin[...,2]+bin[...,3]+bin[...,5]+bin[...,7])%2
            c2=(sign+bin[...,1]+bin[...,2]+bin[...,4]+bin[...,5])%2
            c3=(bin[...,0]+bin[...,1]+bin[...,2]+bin[...,6]+bin[...,7])%2
            c4=(bin[...,3]+bin[...,4]+bin[...,5]+bin[...,6]+bin[...,7])%2
            print (para.shape)
            print (c1.shape)

            a=torch.Tensor([]).to('cuda')
            n_mantissa=abs(mantissa)
            if format=='fp16':
                result_m=torch.zeros(para.shape).half().to('cuda')
            else:
                result_m=torch.zeros(para.shape).to('cuda')
                
            check_count=0
            for i in range (1,index):
                n_mantissa=torch.multiply(n_mantissa,2)
                temp=torch.remainder(n_mantissa.trunc(),2)
                if (i>=index-blind):
                    if format=='fp16':
                        exit()
                        temp=torch.full(temp.shape,0).half().to('cuda')
                    else:
                        #temp=torch.full(temp.shape,0).to('cuda')
                        if check_count==0:
                            temp=c4
                        elif check_count==1:
                            temp=c3
                        elif check_count==2:
                            temp=c2
                        elif check_count==3:
                            temp=c1
                            
                        check_count+=1
                        
                result_m=result_m+temp*(2**-i)
                
            result_m=torch.multiply(result_m,para.sign())
            result_m=torch.ldexp(result_m, exponent)
            if format=='fp16':
                result_m=result_m.half()
            else:
                result_m=result_m
            
            ##############################################################
            if test==True and layer==2:
                for i in range(len(para)):
                    print ("====================================")
                    print ("para:",para[i])
                    mantissa, exponent = torch.frexp(para[i])
                    print ("exp:",exponent)
                    #print ("rest:",result_m[i])
                    bin=dec2bin(exponent+126, bits=8)
                    print ("bin:",bin)
                    print (bin[0]+bin[1]+bin[3])
                    print (abs((para[i].sign()-1)/2))
                    print ("-------------------------")
                    print ("binary_h(para):",binary(para[i]))
                    print ("binary_h(rest):",binary(result_m[i]))
                exit()
            
            if (torch.equal(result_m, para) != True and blind==0):
                print ("error...")
                diff=torch.eq(result_m, para)
                ind=(diff.int() == 0).nonzero(as_tuple=True)
                print (ind)
                for h in range(len(ind[0])):
                    m1, e1 = torch.frexp(result_m[ind][h])
                    m2, e2 = torch.frexp(para[ind][h])
                    count+=1
                    print (m1)
                    print (m2)
                    if format=="fp32":
                        print (binary(result_m[ind][h]))
                        print (binary(para[ind][h]))
                    elif format=="fp16":
                        print (binary_h(result_m[ind][h]))
                        print (binary_h(para[ind][h]))
                        
            #print (result_m)
            model_free.state_dict()[name].data.copy_(result_m)

            
    torch.save(model_free, FILE_half)
    print ("\n\ncount",count)

    
    model = torch.load(FILE_half)
    model = model.to('cuda')
    for layer, (name, para) in enumerate(model.named_parameters()): 
        if layer==1:
            for i in para:
                print ("===================")
                print (i)
                if format=="fp16":
                    print (binary_h(i))
                else:
                    print (binary(i))
        if layer==0:
            para_shape=para.shape
            if len(para_shape) == 4:
                for k in range (para_shape[0]):
                    for c in range (para_shape[1]):
                        for h in range (para_shape[2]):
                            for w in range (para_shape[3]):
                                if format=="fp16":
                                    print (binary_h(model_free.state_dict()[name][k,c,h,w]))
                                else:
                                    bin=str(binary(model_free.state_dict()[name][k,c,h,w]))
                                    print (bin)
                                    a=[*bin]
                                    a=list(map(int, a))
                                    error_bit=31
                                    a[error_bit]=abs(a[error_bit]-1)
                                    c1=a[31]
                                    c2=a[30]
                                    c3=a[29]
                                    c4=a[28]
                                    d1=(a[0]+a[1]+a[3]+a[4]+a[6]+a[8]+c1)%2
                                    d2=(a[0]+a[2]+a[3]+a[5]+a[6]+c2)%2
                                    d3=(a[1]+a[2]+a[3]+a[7]+a[8]+c3)%2
                                    d4=(a[4]+a[5]+a[6]+a[7]+a[8]+c4)%2
                                    print (d4,d3,d2,d1)
            exit()