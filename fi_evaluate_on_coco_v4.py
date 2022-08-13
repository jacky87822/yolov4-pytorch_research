"""
A script to evaluate the model's performance using pre-trained weights using COCO API.
Example usage: python evaluate_on_coco.py -dir D:\cocoDataset\val2017\val2017 -gta D:\cocoDataset\annotatio
ns_trainval2017\annotations\instances_val2017.json -c cfg/yolov4-smaller-input.cfg -g 0
Explanation: set where your images can be found using -dir, then use -gta to point to the ground truth annotations file
and finally -c to point to the config file you want to use to load the network using.
"""

import argparse
import datetime
import json
import logging
import os
os.system(r"rm -rf data/outcome/*.jpg")
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cfg import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names
from tool.torch_utils import do_detect_half,do_detect

import cv2
from tool.utils import plot_boxes_cv2
from datetime import datetime

from torchsummary import summary

from pytorchfi.core import fault_injection
from pytorchfi import extended_func

from operator import add
import protect_method
from fault_inject import inj_model
import csv
import shutil

def get_class_name(cat):
    class_names = load_class_names("./data/coco.names")
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return class_names[cat]

def convert_cat_id_and_reorientate_bbox(single_annotation):
    cat = single_annotation['category_id']
    bbox = single_annotation['bbox']
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h 
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    single_annotation['category_id'] = cat
    single_annotation['bbox'] = [x1, y1, w, h]
    return single_annotation



def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj

def evaluate_on_coco(cfg, resFile, path_name):
    logging.getLogger('PIL').setLevel(logging.WARNING)
    annType = "bbox"  # specify type here
    with open(resFile, 'r') as f:
        unsorted_annotations = json.load(f)
    sorted_annotations = list(sorted(unsorted_annotations, key=lambda single_annotation: single_annotation["image_id"]))
    sorted_annotations = list(map(convert_cat_id_and_reorientate_bbox, sorted_annotations))
    reshaped_annotations = defaultdict(list)
    for annotation in sorted_annotations:
        reshaped_annotations[annotation['image_id']].append(annotation)

    with open(path_name+'/temp.json', 'w') as f:
        json.dump(sorted_annotations, f)

    cocoGt = COCO(cfg.gt_annotations_path)
    cocoDt = cocoGt.loadRes(path_name+'/temp.json')

    with open(cfg.gt_annotations_path, 'r') as f:
        gt_annotation_raw = json.load(f)
        gt_annotation_raw_images = gt_annotation_raw["images"]
        gt_annotation_raw_labels = gt_annotation_raw["annotations"]

    rgb_label = (255, 0, 0)
    rgb_pred = (0, 255, 0)

    for i, image_id in enumerate(reshaped_annotations):
        image_annotations = reshaped_annotations[image_id]
        gt_annotation_image_raw = list(filter(
            lambda image_json: image_json['id'] == image_id, gt_annotation_raw_images
        ))
        gt_annotation_labels_raw = list(filter(
            lambda label_json: label_json['image_id'] == image_id, gt_annotation_raw_labels
        ))
        if len(gt_annotation_image_raw) == 1:
            image_path = os.path.join(cfg.dataset_dir, gt_annotation_image_raw[0]["file_name"])
            actual_image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(actual_image)

            for annotation in image_annotations:
                x1_pred, y1_pred, w, h = annotation['bbox']
                x2_pred, y2_pred = x1_pred + w, y1_pred + h
                cls_id = annotation['category_id']
                label = get_class_name(cls_id)
                draw.text((x1_pred, y1_pred), label, fill=rgb_pred)
                draw.rectangle([x1_pred, y1_pred, x2_pred, y2_pred], outline=rgb_pred)
            for annotation in gt_annotation_labels_raw:
                x1_truth, y1_truth, w, h = annotation['bbox']
                x2_truth, y2_truth = x1_truth + w, y1_truth + h
                cls_id = annotation['category_id']
                label = get_class_name(cls_id)
                draw.text((x1_truth, y1_truth), label, fill=rgb_label)
                draw.rectangle([x1_truth, y1_truth, x2_truth, y2_truth], outline=rgb_label)
            try:
                actual_image.save(path_name+"/outcome/outcomes_{}".format(gt_annotation_image_raw[0]["file_name"]))
            except:
                logging.error ("can't save actual_image")
        else:
            logging.info('please check')
            break
        if (i + 1) % 100 == 0: # just see first 100
            break

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats
    


def test(model, annotations, cfg, path_name):
    model_name=cfg.weights_file.split('/')[-1]

    start = datetime.now()
    if not annotations["images"]:
        print("Annotations do not have 'images' key")
        return
    images = annotations["images"]
    # images = images[:10]
    t0 = datetime.now()
    resFile = path_name+'/coco_val_outputs.json'

    if torch.cuda.is_available():
        use_cuda = 1
    else:
        use_cuda = 0

    #  do one forward pass first to circumvent cold start
    img = cv2.imread('data/dog.jpg')
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    #fi_do_detect(model, inj_list, cfg.inj_mode, sized, 0.001, 0.6, use_cuda)


    if cfg.half:
        boxes = do_detect_half(model, sized, 0.001,  0.6, use_cuda)
    else:
        boxes = do_detect(model, sized, 0.001,  0.6, use_cuda)
    
    boxes_json = []
    res1 = datetime.now()
    res2 = datetime.now()
    time1=res2-res1
    time2=res2-res1
    time3=res2-res1
    time4=res2-res1
    failure_box=0
    failure_img=0
    input_width=model.width
    input_height=model.height

    t0 = time.time()
    for i, image_annotation in enumerate(images):
        t1 = time.time()
        if t1 - t0 > 60*30:
            logging.error('Failed in timeout, current img count:'+ str(i))
            break
        
        if i%100==99:
            logging.info("execute: w: {} b: {} m: {} p: {} s: {}  currently on image: {}/{}".format(model_name, cfg.ber, cfg.inj_mode, 
                                                                                            cfg.protect, cfg.seed,
                                                                                            i + 1, len(images)))
        image_file_name = image_annotation["file_name"]
        image_id = image_annotation["id"]
        image_height = image_annotation["height"]
        image_width = image_annotation["width"]

        if i >1000 and cfg.tmode==True:
            break

        img = cv2.imread(os.path.join(cfg.dataset_dir, image_file_name))
        #print ("image shape:",img.shape,image_height,image_width)
        sized = cv2.resize(img, (input_width, input_height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        
        if cfg.half:
            boxes = do_detect_half(model, sized, 0.001,  0.6, use_cuda)
        else:
            boxes = do_detect(model, sized, 0.001,  0.6, use_cuda)

        if type(boxes) == list:
            for box in boxes[0]:
                try:
                    box_json = {}
                    category_id = box[-1]
                    score = box[-2]
                    box_json["category_id"] = int(category_id)
                    box_json["image_id"] = int(image_id)
                    bbox = []
                    x1 = max(0, int(box[0] * image_width))
                    y1 = max(0, int(box[1] * image_height))
                    x2 = max(0, int(box[2] * image_width))
                    y2 = max(0, int(box[3] * image_height))

                    if x2-x1<0 or y2-y1<0:
                        failure_box+=1
                        logging.error("currently on image: {}/{} id: {}".format(i + 1, len(images),image_id))
                        logging.error("boxes: "+str(boxes))
                        logging.error("box: "+str(box))
                        break

                    bbox.append(x1)
                    bbox.append(y1)
                    bbox.append(x2-x1)
                    bbox.append(y2-y1)
                    box_json["bbox"] = bbox
                    box_json["score"] = round(float(score), 2)
                    boxes_json.append(box_json)
                    # print("see box_json: ", box_json)
                except Exception as e:
                    failure_box+=1
                    #print ("error!!!",image_id)
                    #logging.error("(box) currently on image: {}/{} id: {}".format(i + 1, len(images),image_id))
                    #logging.error('Failed in : '+ str(e))
                    #logging.error("boxes: "+str(boxes))
                    #logging.error("box: "+str(box))
        else:
            #print("warning: output from model after postprocessing is not a list, ignoring")
            return
        namesfile = 'data/coco.names'
        if cfg.output == True or (i + 1) % 100 == 0:
            try:
                class_names = load_class_names(namesfile)
                plot_boxes_cv2(img, boxes[0], path_name+'/predictions/predictions_{}.jpg'.format(image_id), class_names)
            except Exception as e:
                    failure_img+=1
                    #print ("error!!!",image_id)
                    #logging.error("(plot) currently on image: {}/{} id: {}".format(i + 1, len(images),image_id))
                    #logging.error('Failed in : '+ str(e))
        
    with open(resFile, 'w') as outfile:
        json.dump(boxes_json, outfile, default=myconverter)
    try:
        Ap_result=evaluate_on_coco(cfg, resFile, path_name)
    except Exception as e:
        #print ("Error!!! evaluate_on_coco")
        logging.error('Errors in CoCoEval')
        logging.error('Failed in : '+ str(e))
        Ap_result=None
    finish = datetime.now()
    
    print ('Total Duration: {}'.format(finish - start))
    print ('Total failure_box : {}'.format(failure_box))
    print ('Total failure_img : {}'.format(failure_img))
    
    return Ap_result, finish - start, failure_box, failure_img, inj_info

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def summary(path_name,t_op,Ap_result,cfg,seed,inj_info,sum_all):

########################################################################################
    # writeout summary
    logging.info("start writeout summary ...")
    Ap_title= ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] =",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] =",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] =",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] ="]
    
    ########################################################################################
    # writeout summary to txt  
    summary_list=[datetime.now(), cfg.tmode, path_name, cfg.weights_file, layer_num, para_num,
                  in_ber, errbin_num, seed, cfg.inj_mode, cfg.protect,
                  t_op, failure_box, failure_img, maxv.item(), minv.item()]
    print (summary_list)
    
    summary=("======================================================================="+'\n')
    summary+=("Time      : "+str(summary_list[0])+'\n')
    summary+=("Testmode  : "+str(summary_list[1])+'\n')
    summary+=("Result    : "+str(summary_list[2])+'\n')
    summary+=("Weights   : "+str(summary_list[3])+'\n')
    summary+=("#layers   : "+str(summary_list[4])+'\n')
    summary+=("#weights  : "+str(summary_list[5])+'\n')

    summary+=("\n") 
    summary+=("BER       : "+str(summary_list[6])+'\n')
    summary+=("#inject   : "+str(summary_list[7])+'\n')
    summary+=("seed      : "+str(summary_list[8])+'\n')
    summary+=("INJ mode  : "+str(summary_list[9])+'\n')
    summary+=("PRT mode  : "+str(summary_list[10])+'\n')

    summary+=("\n") 
    summary+=("dur_time  : "+str(summary_list[11])+'\n')
    summary+=("#fail img : "+str(summary_list[12])+'\n')
    summary+=("#fail img : "+str(summary_list[13])+'\n')
    
    summary+=("\n") 
    summary+=("max value : "+str(summary_list[14])+'\n')
    summary+=("min value : "+str(summary_list[15])+'\n')
    summary+=("======================================================================="+'\n')
    
    mAP_result=False
    try:
        for index,r in enumerate (Ap_result):
            summary+=("%s %.3f" %(Ap_title[index],r)+'\n')
        mAP_result=True
        r=[Ap_result[0],Ap_result[-3],Ap_result[-2],Ap_result[-1]]
        summary_list.extend(r)
    except:
        summary+=("Error!!! Ap_result: "+str(Ap_result)+'\n')
        mAP_result=False
        summary_list.extend([0,0,0,0])
        
    print (summary_list)
    f=open(path_name+"/summary.txt",'w')
    f.write(summary)
    f.close
    print (summary)
    
    if (sum_all==False) and (org_ber==-1) and (org_seed<0):
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(summary_list)
    elif (sum_all==True) and (org_ber==-1) and (org_seed<0):
        with open(csv_all_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(summary_list)

    return mAP_result

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Test model on test dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', metavar='G', type=str, 
                        default='0',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, 
                        default='/home/wjc111/TARS_Data/2021_Error-mitigation/00_dataset/02_COCO/02_COCO_errorfree/',
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-gta', '--ground_truth_annotations', type=str, 
                        default='./dataset/instances_val2017.json',
                        help='ground truth annotations file', dest='gt_annotations_path')
    parser.add_argument('-w', '--weights_file', type=str, 
                        default='../model/yolov4.pt',
                        help='weights file to load', dest='weights_file')
    parser.add_argument('-t', '--test', type=bool, 
                        default=False,
                        help='test mode', dest='tmode')
    parser.add_argument('-m', '--mode', type=int, 
                        default=1,
                        help='injection mode: (0)error-free (1)bit flip (2)SA0 (3)SA1', 
                        dest='inj_mode')
    parser.add_argument('-b', '--ber', type=int, 
                        default=7,
                        help='Bit Error Rate', dest='ber')
    parser.add_argument('-s', '--seed', type=int, 
                        default=0,
                        help='Random seed', dest='seed')
    parser.add_argument('-o', '--output', type=bool, 
                        default=False,
                        help='output image or not', dest='output')
    parser.add_argument('-p', '--protect', type=float, 
                        default=0,
                        help='(0) None (1)zero-masking(ideal) (2) masked by threshold'+
                        '(3) fix exponent (ideal) (4) TMR exponent (5) parity-zero (6) thres-zero', dest='protect')
    parser.add_argument('-hf', '--half_precision', type=bool, default=False,
                        help='True for inference by fp16', dest='half')    
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)



############################################################################################################
#                                                  MAIN
############################################################################################################


if __name__ == "__main__":
    t_start = datetime.now()
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ########################################################################################
    # csv setting
    header=["Time","Testmode","Result","Weights","#layers","#weights", #5
        "BER","#inject","seed","INJ mode","PRT mode",
        "dur_time","#fail box","#fail img",
        "max","min",
        "mAP","mAP-small","mAP-medium","mAP-large"]

    if cfg.ber==-1 and cfg.seed<0:
        if not os.path.exists("./result/backup_csv"):
            os.makedirs("./result/backup_csv")
        csv_filename='./result/summary.csv'
        now = datetime.now()
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
        else:
            dst='./result/backup_csv/summary_'+str(now).split(".")[0]+".csv"
            shutil.copyfile(csv_filename, dst)
            
        csv_all_filename='./result/summary_all.csv'
        if not os.path.exists(csv_all_filename):
            with open(csv_all_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
        else:
            dst='./result/backup_csv/summary_all_'+str(now).split(".")[0]+".csv"
            shutil.copyfile(csv_all_filename, dst)
    ########################################################################################
    # BER setting
    if 'tiny' in cfg.weights_file:
        default_ber=[0, 1e-7]
    else:
        default_ber=[0, 1e-8]
    #default_ber=[0.004, 0.003, 0.002, 0.001, 0.0007, 0.0005, 0.0003, 0.0001, 0.00001, 0.000001, 0.0000001]
    ber_list=[]
    if cfg.ber==-1:
        ber_list=default_ber
    else:
        ber_list.append(cfg.ber)

    curr_map=-1
    pre_map=-1

    print ("ber_list:",ber_list)
    org_ber=cfg.ber

    ########################################################################################
    # Auto analysis setting
    org_seed= cfg.seed
    if cfg.seed < 0:
        seed_list=range(abs(cfg.seed))
    else:
        seed_list=[cfg.seed]
    

    for iter,curr_ber in enumerate (ber_list):
        ts_start = datetime.now()
        accu_map=[0]*12
        inj_info=[0]*2
        cfg.ber=curr_ber
        ########################################################################################
        # set for top dir
        model_name=cfg.weights_file.split('/')[-1]
        if org_seed < 0:
            top_path="./result/Auto_"+model_name.split('.')[0]+"_fm"+str(cfg.inj_mode)+"_pt"+str(cfg.protect)+"_ber"+str(cfg.ber)+"/"
            if not os.path.exists(top_path):
                os.makedirs(top_path)
                os.makedirs(top_path+"/log")
                logging = init_logger(log_dir=top_path+"/log")
            else:
                print ("auto path "+top_path+" already exists")
                logging = init_logger(log_dir=top_path+"/log")
        else:
            top_path="./result/"

        ########################################################################################
        # *** execute same BER in different seed
        for s_iter, curr_seed in enumerate (seed_list):
            t_start = datetime.now()
            if curr_ber==-1:
                print ("curr_ber==-1")
                exit()
            ########################################################################################
            # Generate storage path
            cfg.seed=curr_seed
            path_name=top_path+model_name.split('.')[0]+"_fm"+str(cfg.inj_mode)+"_pt"+str(cfg.protect)+"_ber"+str(cfg.ber)+"_s"+str(cfg.seed)

            if not os.path.exists(path_name):
                os.makedirs(path_name)
            if not os.path.exists(path_name+"/outcome"):
                os.makedirs(path_name+"/outcome")
            if not os.path.exists(path_name+"/predictions"):
                os.makedirs(path_name+"/predictions")
            if not os.path.exists(path_name+"/log"):
                os.makedirs(path_name+"/log")

            ########################################################################################
            # Generate logging setting
            if org_seed>=0:
                logging = init_logger(log_dir=path_name+"/log")
            logging.info(f'Using device {device}')
            logging.info ('current cfg.ber : {} cfg.seed : {}'.format(cfg.ber,cfg.seed))
            
            ########################################################################################
            # check if summary already exist
            s_exist=False
            if os.path.isfile(path_name+'/summary.txt'):
                s_exist=True
                Ap_result=[]
                num_inj_list=0
                f_s=open(path_name+'/summary.txt','r')
                for c,line in enumerate(f_s):
                    if ("Error!!! Ap_result" in line):
                        Ap_result=None
                        break
                    elif "Average" in line and "@" in line:
                        mAP=line.split('] = ')[-1]
                        Ap_result.append(float(mAP.split('\n')[0]))
                    elif '#inject   : ' in line:
                        num_inj_list=line.split('#inject   : ')[-1]
                        num_inj_list=int(num_inj_list.split('\n')[0])
                f_s.close
                try:
                    logging.info("Ap_result   : "+str(Ap_result))
                    accu_map=list( map(add, accu_map, Ap_result))
                except:
                    logging.info("Ap_result is None"+str(Ap_result))
                    accu_map=list( map(add, accu_map, [0]*12))

                print (accu_map)
            if s_exist==False:
                ########################################################################################
                if cfg.ber>=1.0:
                    in_ber=10**(-cfg.ber)
                else:
                    in_ber=cfg.ber
                # Load model and create injection list   
                logging.info("start Load model and create injection list ...")
                # return model,non_zeros,zeros,layer+1,new_max_value,new_min_value
                if cfg.inj_mode==0:
                    error_mode="free"
                elif cfg.inj_mode==1:
                    error_mode="bf"
                elif cfg.inj_mode==2:
                    error_mode="sa0"
                elif cfg.inj_mode==3:
                    error_mode="sa1"
                model,errbin_num,freebin_num,layer_num,para_num,maxv,minv=inj_model(model_name=cfg.weights_file,
                                                       BER=in_ber,
                                                       error_mode=error_mode,
                                                       protection=cfg.protect,
                                                       test=cfg.tmode,
                                                       device=device)
                #inj_list,check,num, [count_one_fault,count_multi_fault]=extended_func.fi_model(model, device, cfg.inj_mode, BER=in_ber, seed=cfg.seed)
                num_inj_list=errbin_num

                ########################################################################################
                # check annotations file
                logging.info("check annotations file ...")
                annotations_file_path = cfg.gt_annotations_path
                with open(annotations_file_path) as annotations_file:
                    try:
                        annotations = json.load(annotations_file)
                    except:
                        print("annotations file not a json")
                        exit()
            
                ########################################################################################
                # inference and evaluate
                logging.info("start inference and evaluate ...")
                Ap_result, _, failure_box, failure_img, inj_info=test(model=model,
                    annotations=annotations,
                    cfg=cfg, 
                    path_name=path_name)

                t_finish = datetime.now()

                ########################################################################################
                # Write out summary
                t_op=t_finish-t_start
                summary(path_name,t_op,Ap_result,cfg,cfg.seed,inj_info,True)

                ########################################################################################
                # accumulate mAP
                try:
                    logging.info("Ap_result   : "+str(Ap_result))
                    accu_map=list( map(add, accu_map, Ap_result))
                except:
                    logging.info("Ap_result is None"+str(Ap_result))
                    accu_map=list( map(add, accu_map, [0]*12))

            if org_seed<1 and num_inj_list==0:
                break

        ########################################################################################
        # Automatic stop & continue
        logging.info("\n\n\n")
        if org_ber==-1:
            average_map = [i/(s_iter+1) for i in accu_map]
            average_inj_info = [i/(s_iter+1) for i in inj_info]
            curr_map = average_map[0]

            logging.info("pre_map   : "+str(pre_map))
            logging.info("curr_map  : "+str(curr_map))
            logging.info("ber_list  : "+str(ber_list))
            logging.info("curr_ber  : "+str(curr_ber))
            logging.info("=================================")
            logging.info("result check and extend BER list")
            logging.info("=================================")
            if org_ber==-1 and iter==len(ber_list)-1:
                logging.info("org_ber==-1 and iter==len(ber_list)-1")
                if (pre_map==-1):
                    logging.info("...pre_map==-1")
                    ber_list.extend([float(format(curr_ber*10,'.9f'))])
                elif (abs(pre_map-curr_map)/max_map>0.3) or (pre_map-curr_map>0.15):
                    logging.info("...(abs(pre_map-curr_map)/max_map>0.3) or (pre_map-curr_map>0.15)")
                    if curr_map<0.001:
                        logging.info("......curr_map<0.001")
                        ber_list.extend([float(format(curr_ber*0.2, '.9f')), 
                                    float(format(curr_ber*0.4, '.9f')), 
                                    float(format(curr_ber*0.6, '.9f')), 
                                    float(format(curr_ber*0.8, '.9f')),
                                    float(format(-1,'.9f'))]) 
                    elif curr_ber==0.1 and curr_map>=0.001:    #***************
                        logging.info("......curr_ber==0.1 and curr_map>=0.001: ")
                        ber_list.extend([float(format(curr_ber*0.2, '.9f')), 
                                    float(format(curr_ber*0.4, '.9f')), 
                                    float(format(curr_ber*0.6, '.9f')), 
                                    float(format(curr_ber*0.8, '.9f')),
                                    float(format(curr_ber*2, '.9f')), 
                                    #float(format(curr_ber*4, '.9f')), 
                                    #float(format(curr_ber*6, '.9f')), 
                                    float(format(-1, '.9f'))])   
                    else:
                        logging.info("......curr_map>=0.001:")
                        ber_list.extend([float(format(curr_ber*0.2, '.9f')), 
                                    float(format(curr_ber*0.4, '.9f')), 
                                    float(format(curr_ber*0.6, '.9f')), 
                                    float(format(curr_ber*0.8, '.9f')), 
                                    float(format(curr_ber*10, '.9f'))]) 
                elif curr_map>0.001 and curr_ber==0.1:   #***************
                    logging.info("...curr_map>0.001 and curr_ber==0.1:")
                    ber_list.extend([float(format(curr_ber*2, '.9f')), 
                                #float(format(curr_ber*4, '.9f')), 
                                #float(format(curr_ber*6, '.9f')), 
                                float(format(-1, '.9f'))])
                elif curr_map>0.001:
                    logging.info("...curr_map>0.001")
                    ber_list.extend([float(format(curr_ber*10,'.9f'))])
            elif org_ber==-1 and iter!=len(ber_list)-1:
                logging.info("org_ber==-1 and iter!=len(ber_list)-1")
                if curr_map<0.001:
                    logging.info("...curr_map<0.001 exit()")
                    exit()

            logging.info("\n\n")
            logging.info("==================")
            logging.info("ber_list  : "+str(ber_list))
            logging.info("pre_map   : "+str(pre_map))
            logging.info("curr_map  : "+str(curr_map))
            logging.info("==================\n")

            ts_finish = datetime.now()
            ts_op = ts_finish-ts_start
            if org_seed <0 and (os.path.isfile(top_path+'/summary.txt')!=True):
                summary(top_path,ts_op,average_map,cfg,org_seed,average_inj_info,False)

            if pre_map==-1:
                pre_map=curr_map
                max_map=curr_map
            else:
                pre_map=min(curr_map,pre_map)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        seed_list=list( map(add, seed_list, [len(seed_list)]*len(seed_list)))


