import os

target="./result/Auto_TMRs_yolov4_fm1_pt7.1_ber0.0008" 
average_mAP=0
for i in os.listdir(target):
    for j in os.listdir(os.path.join(target,i)):
        if j == "summary.txt":
            file=os.path.join(target,i,j)
            f=open(file,'r')
            fn=f.readlines()
            for line in fn:
                if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =" in line:
                    mAP=line.split("=")[-1]
                    mAP=mAP.split("\n")[0]
                    print (mAP)
                    average_mAP+=float(mAP)
            f.close()
print ("average_mAP:",average_mAP/30)