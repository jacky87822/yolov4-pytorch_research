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

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def dec2bin(x, bits=8):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits=8):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def mantissa_fp2bin(x, bit=23):
    x=torch.abs(x)
    x=torch.multiply(x,2**(bit+1))
    y=dec2bin(x.int(), bits=bit)
    return y

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
        exit()
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

def zero_masking_ideal(para,inject_map,device):
    detect_error=torch.zeros(para.shape).to(device)
    masked_error=torch.zeros(para.shape).to(device)
    for bit in range (32):
        detect_error+=inject_map[...,bit]
    new_para=torch.where(detect_error>0,masked_error,para)

    return new_para

def zero_masking_parity(new_para,inject_map,device):
    detect_error=torch.zeros(new_para.shape).to(device)
    masked_error=torch.zeros(new_para.shape).to(device)
    for bit in range (32):
        detect_error+=inject_map[...,bit]
    new_para=torch.where(detect_error%2==1,masked_error,new_para)

    return new_para

def SECDED_de(inj,device):
    d1=(inj[...,0]+inj[...,1]+inj[...,3]+inj[...,4]+inj[...,6]+inj[...,8]+inj[...,31])%2
    d2=(inj[...,0]+inj[...,2]+inj[...,3]+inj[...,5]+inj[...,6]+inj[...,30])%2
    d3=(inj[...,1]+inj[...,2]+inj[...,3]+inj[...,7]+inj[...,8]+inj[...,29])%2
    d4=(inj[...,4]+inj[...,5]+inj[...,6]+inj[...,7]+inj[...,8]+inj[...,28])%2
    
    masked=torch.ones(inj[...,0].shape).int().to(device)

    inj[...,0]=torch.where(((d4==0)&(d3==0)&(d2==1)&(d1==1)),torch.bitwise_xor(masked,inj[...,0]),inj[...,0])
    inj[...,1]=torch.where(((d4==0)&(d3==1)&(d2==0)&(d1==1)),torch.bitwise_xor(masked,inj[...,1]),inj[...,1])
    inj[...,2]=torch.where(((d4==0)&(d3==1)&(d2==1)&(d1==0)),torch.bitwise_xor(masked,inj[...,2]),inj[...,2])
    inj[...,3]=torch.where(((d4==0)&(d3==1)&(d2==1)&(d1==1)),torch.bitwise_xor(masked,inj[...,3]),inj[...,3])

    inj[...,4]=torch.where(((d4==1)&(d3==0)&(d2==0)&(d1==1)),torch.bitwise_xor(masked,inj[...,4]),inj[...,4])
    inj[...,5]=torch.where(((d4==1)&(d3==0)&(d2==1)&(d1==0)),torch.bitwise_xor(masked,inj[...,5]),inj[...,5])
    inj[...,6]=torch.where(((d4==1)&(d3==0)&(d2==1)&(d1==1)),torch.bitwise_xor(masked,inj[...,6]),inj[...,6])
    inj[...,7]=torch.where(((d4==1)&(d3==1)&(d2==0)&(d1==0)),torch.bitwise_xor(masked,inj[...,7]),inj[...,7])
    inj[...,8]=torch.where(((d4==1)&(d3==1)&(d2==0)&(d1==1)),torch.bitwise_xor(masked,inj[...,8]),inj[...,8])

    return inj

def test_all(test,inject_map_copy,inj_para_copy,para,new_para,non_zeros,zeros,max_exponent,device):
    if (test):
        inject_map_copy=inject_map_copy.flatten(end_dim=-2)
        inj_para_copy=inj_para_copy.flatten(end_dim=-2)

        count=0
        para=para.flatten()
        new_para=new_para.flatten()
        for num,(p,inj) in enumerate (zip(para,new_para)):
            befor=binary(p)
            after=binary(inj)
            befor_list=[*befor]
            after_list=[*after]
            if (after!=befor):
                print ("=========================================")
                print (p,inj)
                ######################################################
                print ("map: ",end="")
                print ("       ",end="")
                for c,a in enumerate (inject_map_copy[num]):
                    if c in [1,9,16,24]:
                        print (r"|",end="")
                    if a==1:
                        print (Fore.BLUE+str(a.int().item()),end="") 
                    else:
                        print (Style.RESET_ALL,end="")
                        print (str(a.int().item()),end="") 
                print (Style.RESET_ALL)  
                
                ######################################################
                print ("ori: ",end="")
                print("{:+.3f}".format(p),end=" ")
                for c,a in enumerate (befor_list):
                    if c in [1,9,16,24]:
                        print (r"|",end="")
                    if a==str(1):
                        print (Fore.RED+str(a),end="") 
                    else:
                        print (Style.RESET_ALL,end="")
                        print (a,end="")  
                print (Style.RESET_ALL)  
                print ("-----------------------------------------")
                ######################################################
                print ("err: ",end="")
                print("{:+.3f}".format(bin2fp(inj_para_copy[num],device=device)),end=" ")
                for c,a in enumerate (inj_para_copy[num]):
                    if c in [1,9,16,24]:
                        print (r"|",end="")
                    if a==1:
                        print (Fore.RED+str(a.int().item()),end="") 
                    else:
                        print (Style.RESET_ALL,end="")
                        print (str(a.int().item()),end="") 
                print (Style.RESET_ALL) 
                ######################################################
                print ("use: ",end="")
                print("{:+.3f}".format(inj),end=" ")
                for c,a in enumerate (after_list):
                    if c in [1,9,16,24]:
                        print (r"|",end="")
                    if a==str(1):
                        print (Fore.RED+str(a),end="") 
                    else:
                        print (Style.RESET_ALL,end="")
                        print (a,end="")  
                print (Style.RESET_ALL)
                print ("\n")
            if inj==0:
                count+=1
        print (zeros,non_zeros,non_zeros/zeros)
        print (max_exponent)
        exit()

def inj_model(model_name,BER,error_mode,protection,device="cuda",choice_device="",test=False,save=""):
    model = torch.load(model_name)
    model=model.to(device)
    start = time.time()    
    zeros=0
    non_zeros=0
    if protection in [6.1,6.2,7,7.1,8.1] or test:
        max_exponent=-5
        max_value=0
        min_value=0
        with torch.no_grad():
            for layer, (name, para) in enumerate(model.named_parameters()): 
                para=para.flatten()
                mantissa, exponent = torch.frexp(para)
                
                cur_max_value=torch.max(para)
                if cur_max_value>max_value:
                    max_value=cur_max_value
                    
                cur_min_value=torch.min(para)
                if cur_min_value<min_value:
                    min_value=cur_min_value
                    
                cur_max_exponent=torch.min(exponent)
                if cur_max_exponent>max_exponent:
                    max_exponent=cur_max_exponent
                
    with torch.no_grad():
        for layer, (name, para) in enumerate(model.named_parameters()): 
            print (name)
            #print (binary(para))
            #print ("--------------------------------")
            mantissa, exponent = torch.frexp(para)
            sign=torch.abs((para.sign()-1)/2)
            exp_bin=dec2bin(exponent+126, bits=8)
            mant_bin=mantissa_fp2bin(mantissa, bit=23)
            
            sign.unsqueeze_(-1)
            para_bin=torch.cat((sign,exp_bin,mant_bin),-1)
            
            #########################################################################
            # inject errors
            if choice_device=="cupy":
                inject_map=torch.as_tensor(cupy.random.choice([0, 1], size=para_bin.shape, p=[1-BER, BER]), device=device)
            else:
                inject_map=torch.from_numpy(np.random.choice([0, 1], size=para_bin.shape, p=[1-BER, BER])).to(device)
            
            non_zeros +=inject_map.nonzero().size(0)
            zeros += inject_map.numel() - non_zeros
            
            # injection
            
            inj_para=inject(para_bin,inject_map,error_mode,device) 
            if test:
                inject_map_copy=inject_map.clone().detach()
                inj_para_copy=inj_para.clone().detach()
            #########################################################################
            # bitwise protection
            
            # ideal zero-masking
            if protection==1:
                new_para=zero_masking_ideal(para,inject_map,device)
                
            # ideal exponent
            elif protection==3:
                inj_para[...,1:9]=para_bin[...,1:9]
                new_para=bin2fp(inj_para,device)
            elif protection==3.1:
                inj_para[...,0:9]=para_bin[...,0:9]
                new_para=bin2fp(inj_para,device)
            elif protection==3.2:
                inj_para[...,0:8]=para_bin[...,0:8]
                new_para=bin2fp(inj_para,device)
            
            # TMR
            elif protection==4:
                v0=torch.bitwise_and(inj_para[...,1:9],inj_para[...,16:24])
                v1=torch.bitwise_and(inj_para[...,1:9],inj_para[...,24:])
                v2=torch.bitwise_and(inj_para[...,16:24],inj_para[...,24:])
                v3=torch.bitwise_or(v0,v1)
                inj_para[...,1:9]=torch.bitwise_or(v3,v2)
                new_para=bin2fp(inj_para,device)
            # TMRs
            elif protection==4.1:
                v0=torch.bitwise_and(inj_para[...,0:8],inj_para[...,16:24])
                v1=torch.bitwise_and(inj_para[...,0:8],inj_para[...,24:])
                v2=torch.bitwise_and(inj_para[...,16:24],inj_para[...,24:])
                v3=torch.bitwise_or(v0,v1)
                inj_para[...,0:8]=torch.bitwise_or(v3,v2)
                new_para=bin2fp(inj_para,device)
                
            # parity-zero masking
            elif protection==5:
                new_para=bin2fp(inj_para,device)
                new_para=zero_masking_parity(new_para,inject_map,device)
                
            # value threshold
            elif protection==6.1:
                new_para=bin2fp(inj_para,device)
                #new_mantissa, new_exponent = torch.frexp(new_para)
                masked=torch.zeros(new_para.shape).to(device)
                new_para=torch.where(((new_para>max_value)|(new_para<min_value)),masked,new_para)
            # exponent threshold
            elif protection==6.2:
                new_para=bin2fp(inj_para,device)
                _, new_exponent = torch.frexp(new_para)
                masked=torch.zeros(new_para.shape).to(device)
                new_para=torch.where((new_exponent>max_exponent),masked,new_para)
                
            # hybrid
            elif protection==7:
                v0=torch.bitwise_and(inj_para[...,0:8],inj_para[...,16:24])
                v1=torch.bitwise_and(inj_para[...,0:8],inj_para[...,24:])
                v2=torch.bitwise_and(inj_para[...,16:24],inj_para[...,24:])
                v3=torch.bitwise_or(v0,v1)
                inj_para[...,0:8]=torch.bitwise_or(v3,v2)
                new_para=bin2fp(inj_para,device)
                masked=torch.zeros(new_para.shape).to(device)
                new_para=torch.where(((new_para>max_value)|(new_para<min_value)),masked,new_para)
                
            elif protection==7.1:
                v0=torch.bitwise_and(inj_para[...,0:8],inj_para[...,16:24])
                v1=torch.bitwise_and(inj_para[...,0:8],inj_para[...,24:])
                v2=torch.bitwise_and(inj_para[...,16:24],inj_para[...,24:])
                v3=torch.bitwise_or(v0,v1)
                inj_para[...,0:8]=torch.bitwise_or(v3,v2)
                new_para=bin2fp(inj_para,device)
                _, new_exponent = torch.frexp(new_para)
                masked=torch.zeros(new_para.shape).to(device)
                new_para=torch.where((new_exponent>max_exponent),masked,new_para)
            
            # SECDED
            elif protection==8:
                inj_para2=SECDED_de(inj_para,device)
                new_para=bin2fp(inj_para2,device)
                
            elif protection==8.1:
                inj_para2=SECDED_de(inj_para,device)
                new_para=bin2fp(inj_para2,device)
                _, new_exponent = torch.frexp(new_para)
                masked=torch.zeros(new_para.shape).to(device)
                new_para=torch.where((new_exponent>max_exponent),masked,new_para)
                
            #########################################################################
            # without protection
            else:
                new_para=bin2fp(inj_para,device)
            torch.cuda.empty_cache()

            #########################################################################
            # check the bin 2 float convert
            if test:
                test_all(test,inject_map_copy,inj_para_copy,para,new_para,non_zeros,zeros,max_exponent,device)
            model.state_dict()[name].data.copy_(new_para)
        if save != "":
            torch.save(model, save)
    end = time.time()  
    print ("cost time:",end-start)
    print ("BER:",(zeros/non_zeros))
    print (zeros,non_zeros)
    
    
    
#inj_model(model_name="../model/yolov4_SECDED.pt",BER=1e-2,error_mode="bf",protection=1,test=False,save="./temp.pt")#,choice_device="cupy")
#
#
#model = torch.load("./temp.pt")
#model=model.to("cpu")
#for layer, (name, para) in enumerate(model.named_parameters()): 
#    para=para.flatten()
#    for p in para:
#        print (binary(p))
#    exit()