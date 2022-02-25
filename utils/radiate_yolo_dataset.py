import os
import numpy as np
import random
from utils.radiate_dataset import RadiateDataset
import utils.radiate_bev_utils as bev_utils
import utils.config as cnf

import torch
import torch.nn.functional as F

import cv2

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class RadiateYOLODataset(RadiateDataset):

    def __init__(self, root_dir, split='train', mode ='TRAIN', data_aug=True, multiscale=False,radar = False, weather = "good"):
        super().__init__(root_dir=root_dir, split=split)
        
        self.weather = weather
        self.split_dir = f"split/{split}_{self.weather}_weather/"
        self.multiscale = multiscale
        self.data_aug = data_aug # TODO
        self.img_size = cnf.BEV_WIDTH
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.radar = radar

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode
        
        self.sample_dir_list = []
        self.sample_idx_list_annot = []
         
        self.scenes = os.listdir(self.split_dir)
        
        if mode == 'TRAIN':
            self.preprocess_yolo_training_data()
        else:
            for scene in self.scenes:
                
                if self.radar:
                    self.sample_dir_list += [f"{self.root_dir}/{scene}/Navtech_Cartesian/{int(sample_id):06d}.png" for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/test_split_radar.txt")]
                    self.sample_idx_list_annot += [(scene, int(sample_id) - 1)  for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/test_split_radar.txt")]
                else:
                    self.sample_dir_list += [f"{self.root_dir}/{scene}/velo_lidar/{int(sample_id):06d}.csv" for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/test_split_lidar.txt")]
                    self.sample_idx_list_annot += [(scene, int(sample_id) - 1) for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/test_split_radar.txt")]
        print(f"Load {mode} samples from {self.root_dir}")
        print(f"Done: total {mode} samples {len(self.sample_dir_list)}")
    
    def preprocess_yolo_training_data(self):
        """
        Discard samples which don't have current training class objects, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        sample_dir_list = []
        sample_idx_list_annot = []
        for scene in self.scenes:
            if self.radar:
                sample_dir_list += [f"{self.root_dir}/{scene}/Navtech_Cartesian/{int(sample_id):06d}.png" for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/train_split_radar.txt")]
                sample_idx_list_annot += [(scene, int(sample_id) - 1) for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/train_split_radar.txt")]
            else:
                sample_dir_list += [f"{self.root_dir}/{scene}/velo_lidar/{int(sample_id):06d}.csv" for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/train_split_lidar.txt")]
                sample_idx_list_annot += [(scene, int(sample_id) - 1) for sample_id in np.loadtxt(f"{self.split_dir}/{scene}/train_split_radar.txt")]
        

        for idx in range(len(sample_dir_list)):
            sample_dir = sample_dir_list[idx]
            sample_annot = sample_idx_list_annot[idx]
            objects = self.get_label(sample_annot)

            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)

            valid_list = []
            for label in labels:
                if int(label[0]) in cnf.CLASS_NAME_TO_ID.values():
                    valid_list.append(label[0])
            if len(valid_list):
                self.sample_dir_list.append(sample_dir)
                self.sample_idx_list_annot.append(sample_annot)

    def __getitem__(self, index):
        sample_dir = self.sample_dir_list[index]
        sample_annot = self.sample_idx_list_annot[index]
        if self.mode in ["TRAIN", "EVAL"]:
                
            objects = self.get_label(sample_annot)   
            
            if self.radar:
                gray_map = self.get_radar(sample_dir)
            else:
                calib = self.get_calib()
                gray_map = self.get_lidar(sample_dir, calib)
            gray_map = cv2.cvtColor(gray_map, cv2.COLOR_BGR2GRAY).T
            gray_map = gray_map.reshape(1,gray_map.shape[0],gray_map.shape[1])
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)

            target = bev_utils.build_yolo_target(labels)

            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
            
            img = torch.from_numpy(gray_map).type(torch.FloatTensor)
            
            if self.data_aug:
                if np.random.random() < 0.5:
                    img, targets = self.horisontal_flip(img, targets)
            return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets
        
    @staticmethod
    def horisontal_flip(images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2] # horizontal flip
        targets[:, 6] = - targets[:, 6] # yaw angle flip

        return images, targets

    def __len__(self):
        return len(self.sample_dir_list)
