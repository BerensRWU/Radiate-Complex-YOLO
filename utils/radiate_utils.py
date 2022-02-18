from __future__ import print_function

import numpy as np
import cv2
import os
import json
import utils.config as cnf
import pandas as pd

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, obj, idx):
        self.type = obj["class_name"] # 'Car', 'Pedestrian', ...
        self.cls_id = self.cls_type_to_id(self.type)

        self.ry = obj["bboxes"][idx]["rotation"] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = np.deg2rad(-self.ry)
        # extract 3d bounding box information
        self.w = obj["bboxes"][idx]["position"][2] # box width
        self.h = obj["bboxes"][idx]["position"][3] # box length (in meters)
        self.t = (obj["bboxes"][idx]["position"][0],obj["bboxes"][idx]["position"][1]) # location (x,y,z) in camera coord.
        #self.dis_to_cam = np.linalg.norm(self.t)
        self.score = -1
            
    def cls_type_to_id(self, cls_type):
        # Car and Van ==> Car class
        # Pedestrian and Person_Sitting ==> Pedestrian Class
        CLASS_NAME_TO_ID = cnf.CLASS_NAME_TO_ID
        if cls_type not in CLASS_NAME_TO_ID.keys():
            return -1
        return CLASS_NAME_TO_ID[cls_type]

    def print_object(self):
        print('Type: %s' % \
            (self.type))
        print('2d bbox h,w: %f, %f' % \
            (self.h, self.w))
        print('2d bbox location, ry: (%f, %f), %f' % \
            (self.t[0],self.t[1],self.ry))
    
    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.type, -1, int(self.occlusion), -1, -1, -1,
                       -1, -1, self.h, self.w, self.l, self.t[0], self.t[1], self.t[2],
                       self.ry, self.score)
        return kitti_str

def read_label(root, scene, idx):
    with open(os.path.join(root, scene, "annotations/annotations.json")) as json_file:
        data = json.load(json_file)
    objects = [Object3d(obj, idx) for obj in data if len(obj["bboxes"][idx])] 
    return objects

def read_lidar(lidar_path):
    """given a lidar raw path returns it lidar point cloud

    :param lidar_path: path to lidar raw point
    :type lidar_path: string
    :return: lidar point cloud Nx5 (x,y,z,intensity,ring)
    :rtype: np.array
    """
    return pd.read_csv(lidar_path, delimiter=',').values

def lidar2radar(lidar, calib):
    M = calib.lidar2radar
    n = lidar.shape[0]
        
    lidar_hom = np.hstack((lidar, np.ones((n,1))))
    lidar_ref = np.dot(lidar_hom, np.transpose(M))
    
    return lidar_ref[:,[0,2,1]]

def lidar_to_image(lidar, calib):
        """Convert an lidar point cloud to an 2d bird's eye view image

        :param lidar: lidar point cloud Nx5 (x,y,z, intensity, ring)
        :type lidar: np.array
        :return: 2d bird's eye image with the lidar information
        :rtype: np.array
        """
        lidar[:,0:3] = lidar2radar(lidar[:,0:3], calib)

        image = np.zeros((1152, 1152, 3))
        h_width = 1152/2.0
        h_height = 1152/2.0
        cell_res_x = 100.0/h_width
        cell_res_y = 100.0/h_height
        for i in range(lidar.shape[0]):
            if False: # Remove Ground
                if lidar[i, 2] > - 1.5: # Ground Threshold
                    image = __inner_lidar_bev_image(
                        lidar, image, i, cell_res_x, cell_res_y, h_width, h_height)
            else:
                image = __inner_lidar_bev_image(
                    lidar, image, i, cell_res_x, cell_res_y, h_width, h_height)
        return image.astype(np.uint8)

def __inner_lidar_bev_image(lidar,
                            image,
                            i,
                            cell_res_x,
                            cell_res_y,
                            h_width,
                            h_height):
        xyzi = lidar[i]
        x = xyzi[0]/cell_res_x + h_width
        y = h_height - xyzi[1]/cell_res_y
        if True: # use LiDAR Channel Number 
            c = int(xyzi[4]) * 8
        else: # Use LiDAR Intensity
            c = int(xyzi[3])
        image = cv2.circle(image, (int(x), int(y)), 1, (c, c, c))
        return image

