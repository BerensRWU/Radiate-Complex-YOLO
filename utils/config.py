import torch
import numpy as np

root_dir = '/home/user/work/radiate/data/'

class_list = ["car", "van", "truck", "bus", "motorbike",
              "bicycle", "pedestrian", "group of pedestrian"]

CLASS_NAME_TO_ID = {
            'car': 		    		0,
            'van': 		    		1,
            'truck': 	    		2,
            'bus': 		    		3,
            'motorbike':    		4,
            'bicycle':   			5,
            'pedestrian':		    6,
            'group of pedestrian': 	7,
            }


BEV_WIDTH = 1152 # Width of the bird's eye view image
BEV_HEIGHT = 1152 # Height of the bird's eye view image

REMOVE_GROUND = False
GROUND_THRESHOLD= -1.5

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
