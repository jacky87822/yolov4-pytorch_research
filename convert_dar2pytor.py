import torch
import torch.nn as nn
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import os
name_list=['yolov4','yolov3','yolov4-tiny','yolov3-tiny']

for i in name_list:
    print ("convert:",i,'...')
    cfgfile="../model/"+i+".cfg"
    weightfile="../model/"+i+".weights"
    FILE="../model/"+i+".pt"

    WEIGHTS = Darknet(cfgfile)
    WEIGHTS.load_weights(weightfile)

    torch.save(WEIGHTS, FILE)

    print ("check model")
    model = torch.load(FILE)
    model = model.to('cuda')
    f=open("../model/pytorch_summary_"+i+".txt",'w')
    if "tiny" in i:
        summary(model, (3, 416, 416))
        f.write(str(summary(model, (3, 416, 416))))
    else:
        summary(model, (3, 416, 416))
        f.write(str(summary(model, (3, 416, 416))))
    print ("\n\n")
    
    
    f.close()