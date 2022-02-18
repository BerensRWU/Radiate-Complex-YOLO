from __future__ import division

from models import *
from utils.utils import *

import os, sys, time, datetime, argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import utils.config as cnf
from utils.radiate_yolo_dataset import RadiateYOLODataset

import matplotlib.pyplot as plt
import numpy as np
def evaluate(model, iou_thres, conf_thres, nms_thres, img_size, batch_size, weather):
    model.eval()

    # Get dataloader
    split='valid'
    dataset = RadiateYOLODataset(cnf.root_dir, split=split, mode='EVAL', data_aug=False, radar = opt.radar, weather = weather)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        # Extract labels
        labels += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        
        sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--radar", default=False, action='store_true' , help="Use Radar Data")
    parser.add_argument("--weather", default = "all", type = str, help = "Choose weather conditions: all (default), good, bad(rain,fog,snow), rain")
    opt = parser.parse_args()
    print(opt)

    sensor = "radar" if opt.radar else "lidar"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AP_list = []
    check_points = os.listdir(f"checkpoints_{sensor}/")
    try:
        for check_point in check_points:
            # Initiate model
            model = Darknet(opt.model_def).to(device)
            # Load checkpoint weights
            model.load_state_dict(torch.load(f"checkpoints_{sensor}/{check_point}", map_location = device))
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                weather = opt.weather,
            )

            AP_list.append(AP)
    finally:
        AP_list = np.array(AP_list)
        
        np.save(f"{sensor}", AP_list)
        plt.plot(np.arange(0, AP_list.shape[0]*10,10), AP_list[:,0])
        plt.xlabel("Epoch")
        plt.ylabel("AP")
        plt.title(f"{sensor} Data Radiate")
        plt.savefig(f"{sensor}.png")
    
    
