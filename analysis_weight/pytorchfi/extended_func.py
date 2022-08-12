from pytorchfi.core import fault_injection
from torchsummary import summary
import torch
import numpy as np
import struct
import matplotlib
matplotlib.use('agg')
import random
from matplotlib import pyplot as plt
import seaborn as sns
plt.ion()
import logging

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

def num_ctrl_weights(pfi_model):
    count=0
    layer_num=0
    for layer, _dim in enumerate(pfi_model.weights_size):
        temp=1
        for i in _dim:
            temp=temp*i
        count=count+temp
        layer_num=layer
    return layer_num,count

def errorrate_inj(model,num_ctrl_weights,BER=0.0000001,seed=0):
    """
    function create_inj_list return required injection info for pytorchfi
    currently support BER in weights for bit-flip, SAF and also masking to zero 
        model : pytorch model
        num_ctrl_weights : number of weights under search
        BER : bit error rate default in 0.000001
        seed : random seed, the result will be the same under same seed number
    the return value will be a list contains follows:
        ayer, k, C, H, W, err_val, #of errors in weight
        Note: err_val contain [0] error-free [1] single bit flip [2] SA0 [3] SA1 [4] masking to 0
        check: any problem or not
    """
    print ("\n")
    print ("==========================")
    print ("seed:",seed,"\nBER:",BER,"\ninject num:",int(BER*num_ctrl_weights))
    print ("==========================")
    #np.random.seed(seed)
    logging.info('random position choice')
    #print ("model.parameters()",model.parameters())
    #inj_list=np.random.randint(num_ctrl_weights*32, size=int(BER*num_ctrl_weights))
    #inj_list=np.random.choice((num_ctrl_weights*32), int(BER*num_ctrl_weights), replace=False)
    inj_list=np.random.randint(num_ctrl_weights*32, size=int(BER*num_ctrl_weights))
    while (True):
        inj_list=np.unique(inj_list) 
        logging.info("inj_list: "+str(len(inj_list)))
        if len(inj_list)>=int(BER*num_ctrl_weights):
            np.random.shuffle(inj_list) 
            break
        else:
            inj_list=np.append(inj_list,np.random.randint(num_ctrl_weights*32, size=int(BER*num_ctrl_weights)))
            
    inj_list=inj_list[:int(BER*num_ctrl_weights)]
        
    inj_list.sort()
    logging.info('start create inject list')
    check=True
    scan_inj_index=0
    scan_bit_count=0
    ac_layer=0
    pre_inj_word_count=None
    inj_result=[]

    for layer, (name, para) in enumerate(model.named_parameters()): 
        inj_bit_loc_list=[]
        if scan_inj_index==len(inj_list):
            break
        if "bias" not in name:    
            #print (name)    
            flatten=para.flatten()
            layer_shape=para.shape  
            while (inj_list[scan_inj_index]<(len(flatten)*32)+scan_bit_count):
                inj_word_count=int((inj_list[scan_inj_index]-scan_bit_count)/32)
                inj_bit_loc=(inj_list[scan_inj_index]-scan_bit_count)%32
                if inj_word_count==pre_inj_word_count:
                    #print ("\n\nerrors in same weights !!!\n\n")
                    #print ("original inj_result",len(inj_result))
                    inj_result.pop()
                    #print ("removed inj_result",len(inj_result))
                else:
                    inj_bit_loc_list=[]
                inj_bit_loc_list.append(inj_bit_loc)

                index=[]
                temp=inj_word_count
                for c in range(len(layer_shape)):
                    index.append(temp%layer_shape[len(layer_shape)-c-1])
                    temp=int(temp/layer_shape[len(layer_shape)-c-1])
                index.reverse()
                
                value=para
                for i in index:
                    value=value[i]
                if (value !=flatten[inj_word_count]):
                    print ("error!!!",index,inj_word_count)
                    check=False
                if len(index)!=4:
                    for i in range(4-len(index)):
                        index.insert(0,0)
                if len(layer_shape) != 4:
                    index[0]=None # k=none
                    index[2]=None # h=none
                    index[3]=None # w=none
                

                index.append(inj_bit_loc_list)

                value=float(value)
                index.insert(0,ac_layer)
                fp=str(binary(value))
                fp_split=[*fp]

                bfp_value_list=fp_split.copy()
                sao_value_list=fp_split.copy()
                sa1_value_list=fp_split.copy()
                for bit_error in inj_bit_loc_list:
                    bfp_value_list[bit_error]=str(1-int(fp_split[bit_error])) #bit flip
                    bfp_value = ''.join(bfp_value_list)
                    sao_value_list[bit_error]=str(0) #SA0
                    sao_value = ''.join(sao_value_list)
                    sa1_value_list[bit_error]=str(1) #SA1
                    sa1_value = ''.join(sa1_value_list)

                    #if len(inj_bit_loc_list)>1:
                    #    print ("\nori:",fp,"\nbfp:",bfp_value,"\nsao:",sao_value,"\nsa1:",sa1_value)


                bfp_value=ieee_754_conversion(int(bfp_value,2))
                sao_value=ieee_754_conversion(int(sao_value,2))
                sa1_value=ieee_754_conversion(int(sa1_value,2))

                masked=0
                
                # modified there
                '''
                if abs(bfp_value) > 4096:
                    fix_value = 4096
                else:
                    fix_value = bfp_value
                '''
                index.append([value, bfp_value, sao_value, sa1_value])
                index.append(len(inj_bit_loc_list))
                #if (value !=sao_value and value !=sa1_value):
                #    check=False

                inj_result.append(index)
                #print('{}/{} Injecting...: {}\n'.format(scan_inj_index+1,int(BER*num_ctrl_weights),index))
                fp=str(binary(value))

                # set next turn
                pre_inj_word_count=inj_word_count
                scan_inj_index+=1
                if scan_inj_index==len(inj_list):
                    break
            if layer==0:
                #print (list(para))
                pass
            scan_bit_count+=len(flatten)*32
            ac_layer+=1

    print ("finish create injection list !!!")
    return inj_result,check

def inject_summary(inj_result,pfi_model,layer_num):
    layer_num=layer_num+1
    print ("layer_num:",layer_num)
    faulty_layer=np.zeros(layer_num)
    faulty_bit=np.zeros(32)
    count_one_fault=0
    count_multi_fault=0

    for i in inj_result:
        #print (i)
        layer=i[0]
        batch=(i[1]==None)
        bit=i[5]
        inject_info=i[6]
        num_error_bit=i[7]

        faulty_layer[layer]+=1
        for h in bit:
            faulty_bit[h]+=1
        if num_error_bit==1:
            count_one_fault+=1
        else:
            count_multi_fault+=1
    print ("one fault:",count_one_fault)
    print ("multiple fault:",count_multi_fault)
    print ("\n\n")
    '''
    fig_type='linear' #log

    fig = plt.figure(dpi=1000)
    
    fig = sns.set_style('darkgrid')
    fig = sns.set_context(rc={'lines.linewidth': 3.0})
    fig = sns.lineplot(y=faulty_layer, x=np.array(range(0,layer_num))) 

    fig = plt.yscale(fig_type)
    fig = plt.title("Location of Faulty Weights")
    fig = plt.ylabel("Count")
    fig = plt.xlabel("layer")
    #fig.figure.savefig(r'./test_out1.jpg', dpi=1000) 
    plt.show()

    fig2 = plt.figure(dpi=1000)
    #fig = sns.displot(value)
    fig2 = sns.set_style('darkgrid')
    fig2 = sns.set_context(rc={'lines.linewidth': 3.0})
    fig2 = sns.lineplot(y=faulty_bit, x=np.array(range(31,-1,-1)))  
       
    fig2 = plt.yscale(fig_type)  
    fig2 = plt.title("Location of Faulty Bit")
    fig2 = plt.ylabel("Count")
    fig2 = plt.xlabel("Bit Position")
    #fig.figure.savefig(r'./test_out1.jpg', dpi=1000) 
    plt.show()
    '''
    return [count_one_fault,count_multi_fault]




def update_fimodel(pfi_model,inj_result,mode=1):
    #inj_result=[[0,0,0,0,0,[0,0,0]]]
    print ("With",len(inj_result),"faulty weights")
    layer, k, C, H, W, err_val =[],[],[],[],[],[]
    for i in inj_result:
        layer.append(i[0])
        k.append(i[1])
        C.append(i[2])
        H.append(i[3])
        W.append(i[4])
        err_val.append(i[5][mode]) #[0] error-free [1] single bit flip [2] SA0 [3] SA1 [4] masking to 0
    inj = pfi_model.declare_weight_fi(k=k, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val)
    print ("finish update injected model...")
    return inj

def fi_model(model, device, mode, BER=0.0000001, seed=0):
    #summary(model, (3, model.width, model.height))
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    pfi_model = fault_injection(model, 
    1,
    input_shape=[3,model.height,model.width],
    layer_types=[torch.nn.Conv2d,torch.nn.BatchNorm2d],
    use_cuda=use_cuda,
    )

    layer_num,num=num_ctrl_weights(pfi_model)
    inj_result,check=errorrate_inj(model,num,BER=BER,seed=seed)
    pfi_model.print_pytorchfi_layer_summary()

    print ("\n")
    print ("==========================")
    print ("BER     :",BER)
    print ("#inject :",len(inj_result),'/',num)
    print ("seed    :",seed)
    print ("check   :",check)
    print ("mode    :",mode)
    print ("==========================")

    [count_one_fault,count_multi_fault]=inject_summary(inj_result,pfi_model,layer_num)

    return inj_result,check,[layer_num,num], [count_one_fault,count_multi_fault]