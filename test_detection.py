import numpy as np
import math
import os
import argparse
import time
import torch
import cv2

import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.radiate_bev_utils as bev_utils
from utils.radiate_yolo_dataset import RadiateYOLODataset
import utils.config as cnf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_radar/yolov3_ckpt_epoch-290.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--split", type=str, default="test", help="text file having image lists in dataset")
    parser.add_argument("--radar", default=False, action='store_true' , help="Use Radar Data")
    parser.add_argument("--weather", default = "good", type = str, help = "Choose weather conditions: good(default), good_and_bad, bad")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor = "radar" if opt.radar else "lidar"

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path, map_location = device))
    # Eval mode
    model.eval()
    
    dataset = RadiateYOLODataset(cnf.root_dir, split=opt.split, mode='EVAL', data_aug=False, radar=opt.radar, weather = opt.weather)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    #Tensor = torch.FloatTensor

    start_time = time.time()                        
    for index, (bev_maps, targets) in enumerate(data_loader):
        targets = targets[0]
        targets[:, 2:] *= opt.img_size
        # Configure bev image
        input_imgs = Variable(bev_maps.type(Tensor))

        # Get detections 
        with torch.no_grad():
            detections = model(input_imgs)
            detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres) 
        
        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)
        bev_maps = torch.squeeze(bev_maps).numpy()
        bev_maps = cv2.cvtColor(bev_maps, cv2.COLOR_GRAY2BGR)
        #RGB_Map = bev_maps.copy()
        #RGB_Map = RGB_Map.astype(np.uint8)
        
        for _,cls,x,y,w,l,im,re in targets:
            yaw = np.arctan2(im,re)
            bev_utils.drawRotatedBox(bev_maps, x, y, w, l, yaw, [0, 255, 0])
        print(img_detections[0].shape, index)
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, opt.img_size, bev_maps.shape[:2])
            
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                bev_utils.drawRotatedBox(bev_maps, x, y, w, l, yaw, [0,0,255])
        
        cv2.imwrite(f"output_{sensor}/{index: 06d}.png", bev_maps)
