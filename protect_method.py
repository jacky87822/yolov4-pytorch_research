import struct
import torch
def SECDED_de(a):
    a=list(map(int, a))
    c1=a[31]
    c2=a[30]
    c3=a[29]
    c4=a[28]
    d1=(a[0]+a[1]+a[3]+a[4]+a[6]+a[8]+c1)%2
    d2=(a[0]+a[2]+a[3]+a[5]+a[6]+c2)%2
    d3=(a[1]+a[2]+a[3]+a[7]+a[8]+c3)%2
    d4=(a[4]+a[5]+a[6]+a[7]+a[8]+c4)%2
    if (d4,d3,d2,d1)==(0,0,1,1):
        a[0]=abs(a[0]-1)
    elif (d4,d3,d2,d1)==(0,1,0,1):
        a[1]=abs(a[1]-1)
    elif (d4,d3,d2,d1)==(0,1,1,0):
        a[2]=abs(a[2]-1)
    elif (d4,d3,d2,d1)==(0,1,1,1):
        a[3]=abs(a[3]-1)
    elif (d4,d3,d2,d1)==(1,0,0,1):
        a[4]=abs(a[4]-1)
    elif (d4,d3,d2,d1)==(1,0,1,0):
        a[5]=abs(a[5]-1)
    elif (d4,d3,d2,d1)==(1,0,1,1):
        a[6]=abs(a[6]-1)
    elif (d4,d3,d2,d1)==(1,1,0,0):
        a[7]=abs(a[7]-1)
    elif (d4,d3,d2,d1)==(1,1,0,1):
        a[8]=abs(a[8]-1)
    a=list(map(str, a))
    return a
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


def protect_method(i,cfg,bn_layer):
    # no protect
    if cfg.protect==0: 
        inject_value=i[6][cfg.inj_mode]

    # zero-masking (ideal)
    elif cfg.protect==1: 
        inject_value=0

    # Fixed value (ideal) but actually its same as using Threshold to detect
    elif cfg.protect==2:
        exit()
        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if i[0] not in bn_layer:
                if i[6][cfg.inj_mode] > 3.414:
                    inject_value = 3.414
                elif i[6][cfg.inj_mode] < -8.808:
                    inject_value = -8.808
                else:
                    inject_value = i[6][cfg.inj_mode]
            else:
                if i[6][cfg.inj_mode] > 3294.224:
                    inject_value = 3294
                elif i[6][cfg.inj_mode] < -52.33:
                    inject_value = -52.33
                else:
                    inject_value = i[6][cfg.inj_mode]
        else:
            exit()

    # error free exponent
    elif cfg.protect==3: # only exponent
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]

        free_value=str(binary(i[6][0]))
        free_value_split=[*free_value]

        inject_value_list=([faulty_value_split[0]]+free_value_split[1:9])+faulty_value_split[9:]

        inject_value = ''.join(inject_value_list)
        inject_value=ieee_754_conversion(int(inject_value,2))
        
    elif cfg.protect==3.1: # sign bit and exponent
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]

        free_value=str(binary(i[6][0]))
        free_value_split=[*free_value]

        inject_value_list=free_value_split[:9]+faulty_value_split[9:]

        inject_value = ''.join(inject_value_list)
        inject_value=ieee_754_conversion(int(inject_value,2))
    elif cfg.protect==3.2: # sign bit and exponent[7:1]
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]

        free_value=str(binary(i[6][0]))
        free_value_split=[*free_value]

        inject_value_list=free_value_split[:8]+faulty_value_split[8:]

        inject_value = ''.join(inject_value_list)
        inject_value=ieee_754_conversion(int(inject_value,2))
        
    # TMR
    elif cfg.protect==4:
        if "TMR" not in cfg.weights_file:
            exit()
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]
        for bit in range (1,9):
            a=faulty_value_split[bit]
            b=faulty_value_split[bit+15]
            c=faulty_value_split[bit+23]
            faulty_value_split[bit]=str((int(a)&int(b))^(int(a)&int(c))^(int(b)&int(c)))
        inject_value = ''.join(faulty_value_split)
        inject_value=ieee_754_conversion(int(inject_value,2))

    elif cfg.protect==4.1:
        if "TMRs" not in cfg.weights_file:
            exit()
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]
        for bit in range (0,8):
            a=faulty_value_split[bit]
            b=faulty_value_split[bit+16]
            c=faulty_value_split[bit+24]
            faulty_value_split[bit]=str((int(a)&int(b))^(int(a)&int(c))^(int(b)&int(c)))
        inject_value = ''.join(faulty_value_split)
        inject_value=ieee_754_conversion(int(inject_value,2))
        
    # OP
    elif cfg.protect==5:
        if int(i[7]) % 2 == 1:
            inject_value = 0
        else:
            inject_value = i[6][cfg.inj_mode]
            
    # threshold
    elif cfg.protect==6:
        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if i[0] not in bn_layer:
                if i[6][cfg.inj_mode] > 3.414:
                    inject_value = 0
                elif i[6][cfg.inj_mode] < -8.808:
                    inject_value = 0
                else:
                    inject_value = i[6][cfg.inj_mode]
            else:
                if i[6][cfg.inj_mode] > 7.74:
                    inject_value = 0
                elif i[6][cfg.inj_mode] < -12.12:
                    inject_value = 0
                else:
                    inject_value = i[6][cfg.inj_mode]
        else:
            exit()

    elif cfg.protect==6.1:
        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if i[6][cfg.inj_mode] > 7.74:
                inject_value = 0
            elif i[6][cfg.inj_mode] < -12.12:
                inject_value = 0
            else:
                inject_value = i[6][cfg.inj_mode]
        else:
            exit()
    elif cfg.protect==6.2:
        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if i[6][cfg.inj_mode] > 16:
                inject_value = 0
            elif i[6][cfg.inj_mode] < -16:
                inject_value = 0
            else:
                inject_value = i[6][cfg.inj_mode]
        else:
            exit()

    # proposed
    elif cfg.protect==7:
        if "TMRs" not in cfg.weights_file:
            print (cfg.weights_file)
            exit()
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]
        for bit in range (0,8):
            a=faulty_value_split[bit]
            b=faulty_value_split[bit+16]
            c=faulty_value_split[bit+24]
            faulty_value_split[bit]=str((int(a)&int(b))^(int(a)&int(c))^(int(b)&int(c)))
        inject_value = ''.join(faulty_value_split)
        inject_value=ieee_754_conversion(int(inject_value,2))

        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if inject_value > 7.74:
                inject_value = 0
            elif inject_value < -12.12:
                inject_value = 0
            else:
                inject_value = inject_value
        else:
            exit()
            
    elif cfg.protect==7.1:
        if "TMRs" not in cfg.weights_file:
            print (cfg.weights_file)
            exit()
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]
        for bit in range (0,8):
            a=faulty_value_split[bit]
            b=faulty_value_split[bit+16]
            c=faulty_value_split[bit+24]
            faulty_value_split[bit]=str((int(a)&int(b))^(int(a)&int(c))^(int(b)&int(c)))
        inject_value = ''.join(faulty_value_split)
        inject_value=ieee_754_conversion(int(inject_value,2))

        if "yolov4" in cfg.weights_file and "tiny" not in cfg.weights_file:
            if inject_value >= 16:
                inject_value = 0
            elif inject_value <= -16:
                inject_value = 0
            else:
                inject_value = inject_value
        else:
            exit()
        
    # SECDED
    elif cfg.protect==8:
        if "SECDED" not in cfg.weights_file:
            print (cfg.weights_file)
            exit()
        faulty_value=str(binary(i[6][cfg.inj_mode]))
        faulty_value_split=[*faulty_value]
        faulty_value_split=SECDED_de(faulty_value_split)
        inject_value = ''.join(faulty_value_split)
        inject_value=ieee_754_conversion(int(inject_value,2))
        
        
    # fault free
    elif cfg.protect==-1: # fault free
        inject_value=i[6][0]

    return inject_value
    ########################################################################
    

def MSE(error_free_list,without_p_list,with_p_list,weight_num,device):
    MSE_wo_value=0
    MSE_wp_value=0
    max_value=0
    min_value=0
    if len(without_p_list) !=0:
        without_p_list=torch.tensor(without_p_list)
        error_free_list=torch.tensor(error_free_list)
        with_p_list=torch.tensor(with_p_list)
        without_p_list = without_p_list.double().to(device)
        error_free_list = error_free_list.double().to(device)
        with_p_list = with_p_list.double().to(device)
        error_wo=torch.sub(without_p_list,error_free_list)
        SE_wo=torch.square(error_wo)
        MSE_wo=torch.div(SE_wo,weight_num)
        MSE_wo=torch.cumsum(MSE_wo, dim=0)
        error_wp=torch.sub(with_p_list,error_free_list)
        SE_wp=torch.square(error_wp)
        MSE_wp=torch.div(SE_wp,weight_num)
        MSE_wp=torch.cumsum(MSE_wp, dim=0)
        #print (MSE_wo)
        #print (MSE_wp)
        #print (MSE_wo[-1])
        #print (MSE_wp[-1])
        MSE_wo_value=MSE_wo[-1].item()
        MSE_wp_value=MSE_wp[-1].item()
        max_value=torch.max(with_p_list).item()
        min_value=torch.min(with_p_list).item()
    
    return [MSE_wo_value,MSE_wp_value,max_value,min_value]