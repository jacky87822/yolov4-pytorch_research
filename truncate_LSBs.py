import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
import numpy as np
import struct

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
def binary_h(num):
    return bin(np.float16(num.cpu().detach().numpy()).view('H'))[2:].zfill(16)

blind_list=[5,10,16,17,18,19,20,21,22,23]

for blind in blind_list:
    test=False
    i = 'yolov4'
    sa=0
    FILE="../model/"+i+".pt"
    FILE_half="../model/half_"+i+"_b"+str(sa)+"_"+str(blind)+".pt"
    format="fp32"
    if format=="fp32":
        FILE_half="../model/full_"+i+"_b"+str(sa)+"_"+str(blind)+".pt"


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
            a=torch.Tensor([]).to('cuda')
            n_mantissa=abs(mantissa)
            if format=='fp16':
                result_m=torch.zeros(para.shape).half().to('cuda')
            else:
                result_m=torch.zeros(para.shape).to('cuda')
                
            for i in range (1,index):
                n_mantissa=torch.multiply(n_mantissa,2)
                temp=torch.remainder(n_mantissa.trunc(),2)
                if (i>=index-blind):
                    if format=='fp16':
                        temp=torch.full(temp.shape,int(sa)).half().to('cuda')
                    else:
                        temp=torch.full(temp.shape,int(sa)).to('cuda')
                        
                result_m=result_m+temp*(2**-i)
                
            result_m=torch.multiply(result_m,para.sign())
            result_m=torch.ldexp(result_m, exponent)
            if format=='fp16':
                result_m=result_m.half()
            else:
                result_m=result_m
                
            if test==True and layer==2:
                for i in range(len(para)):
                    print ("====================================")
                    print ("para:",para[i])
                    print ("rest:",result_m[i])
                    print ("-------------------------")
                    print ("binary_h(para):",binary_h(para[i]))
                    print ("binary_h(rest):",binary_h(result_m[i]))
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
                                    print (binary(model_free.state_dict()[name][k,c,h,w]))

