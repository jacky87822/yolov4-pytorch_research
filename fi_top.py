import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip',
                    choices=['205', '139', '194', '129', '204', 'all'] ,
                    default='--',
                    help='ServerIP', 
                    dest='ip')
args = parser.parse_args()


#205
if args.ip=='205' or args.ip=='all':
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 0 ")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 2 -p 0 ")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 3 -p 0 ")
#139
if args.ip=='139' or args.ip=='all':
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 1")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 3")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 3.1")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 3.2")
#194
if args.ip=='194' or args.ip=='all':
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 4 -w ../model/TMR_yolov4.pt")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 4.1 -w ../model/TMRs_yolov4.pt")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 5")
#204
if args.ip=='204' or args.ip=='all':
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 6.1")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 6.2")
#129
if args.ip=='129' or args.ip=='all':
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 7 -w ../model/TMRs_yolov4.pt")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 7.1 -w ../model/TMRs_yolov4.pt")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 8 -w ../model/yolov4_SECDED.pt")
    os.system("python fi_evaluate_on_coco_v4.py -s -30 -b -1 -m 1 -p 8.1 -w ../model/yolov4_SECDED.pt")
