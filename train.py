from __future__ import division

from models import *
from utils.utils import *
from utils.radiate_yolo_dataset import RadiateYOLODataset
import utils.config as cnf

from terminaltables import AsciiTable
import os, sys, time, datetime, argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import cv2

nl = '\n'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 301, help = "number of epochs")
    parser.add_argument("--batch_size", type = int, default = 4, help = "size of each image batch")
    parser.add_argument("--gradient_accumulations", type = int, default = 2, help = "number of gradient accums before step")
    parser.add_argument("--model_def", type = str, default = "config/yolov3-custom.cfg", help = "path to model definition file")
    parser.add_argument("--pretrained_weights", type = str, help = "if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type = int, default = 8, help = "number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type = int, default = cnf.BEV_WIDTH, help = "size of each image dimension")
    parser.add_argument("--save_interval", type = int, default = 10, help = "interval to save the weights")
    parser.add_argument("--multiscale_training" ,default = True, type = int, help = "allow for multi-scale training")
    parser.add_argument("--radar", default = False, action = 'store_true' , help = "Use Radar Data")
    parser.add_argument("--weather", default = "good", type = str, help = "Choose weather conditions: good(default), good_and_bad, bad")
    opt = parser.parse_args()
    print(opt)
    
    sensor = "radar" if opt.radar else "lidar"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"checkpoints_{sensor}_{opt.weather}", exist_ok=True)

    # Initiate model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    # Get dataloader
    
    dataset = RadiateYOLODataset(
        cnf.root_dir,
        split = 'train',
        mode = 'TRAIN',
        data_aug = False,#TODO
        multiscale = bool(opt.multiscale_training),
        radar = opt.radar,
        weather = opt.weather
    )

    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "im",
        "re",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    max_mAP = 0.0
    for epoch in range(0, opt.epochs, 1):
        model.train()
        start_time = time.time()
        
        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            loss.backward()
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = f"{nl}---- [Epoch {epoch}/{opt.epochs}, Batch {batch_i}/{len(dataloader)}] ----{nl}"

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"{nl}Total loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"{nl}---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
        
        if epoch % opt.save_interval == 0:
            torch.save(model.state_dict(), f"checkpoints_{sensor}_{opt.weather}/yolov3_ckpt_epoch-{epoch}.pth")
            
        
