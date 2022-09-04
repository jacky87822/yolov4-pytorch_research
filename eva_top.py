import os
# truncate
#os.system("python evaluate_on_coco.py -w ../model/yolov4_trunc_32_global.pt")
#os.system("python evaluate_on_coco.py -w ../model/yolov4_trunc_16_global.pt")
#os.system("python evaluate_on_coco.py -w ../model/yolov4_trunc_8_global.pt")
##os.system("python evaluate_on_coco.py -w ../model/yolov4_trunc_4.pt")

# remove LSBs

'''
blind_list=[5,10,16,17,18,19,20,21,22,23]
name = 'yolov4'
for i in blind_list:
    FILE_half="../model/full_"+name+"_blind_"+str(i)+".pt"
    os.system("python evaluate_on_coco.py -w "+FILE_half)
os.system("python evaluate_on_coco.py -w ../model/yolov4.pt")

blind_list=[5,10,16,17,18,19,20,21,22,23]
name = 'yolov4-tiny'
for i in blind_list:
    FILE_half="../model/full_"+name+"_blind_"+str(i)+".pt"
    os.system("python evaluate_on_coco.py -w "+FILE_half)
os.system("python evaluate_on_coco.py -w ../model/yolov4-tiny.pt")
'''

sa_list=[1,0]
for sa in sa_list:
    blind_list=[5,10,16,17,18,19,20,21,22,23]
    name = 'yolov4'
    for i in blind_list:
        FILE_half="../model/full_"+str(name)+"_b"+str(sa)+"_"+str(i)+".pt"
        os.system("python evaluate_on_coco.py -w "+FILE_half)
os.system("python evaluate_on_coco.py -w ../model/yolov4.pt")


