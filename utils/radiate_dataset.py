from __future__ import division
import glob, os
import numpy as np
import cv2
import torch.utils.data as torch_data
import yaml
import utils.radiate_utils as radiate_utils
from utils.calibration import Calibration

class RadiateDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
                
    def get_radar(self, sample_dir):
        assert os.path.exists(sample_dir), sample_dir
        radar_cartesian = cv2.imread(sample_dir)
        return radar_cartesian
        
    def get_lidar(self, sample_dir, calib):
        assert os.path.exists(sample_dir), sample_dir
        lidar = radiate_utils.read_lidar(sample_dir)
        lidar = radiate_utils.lidar_to_image(lidar, calib)
        return lidar

    def get_calib(self):
        with open("config/default-calib.yaml", 'r') as file:
            calib = yaml.full_load(file)
        # generate calibration matrices from calib file
        calib = Calibration(calib)
        return calib

    def get_label(self, sample_annot):
        scene = sample_annot[0]
        idx = sample_annot[1]
        objects = radiate_utils.read_label(self.root_dir, scene, idx)
        return objects

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
