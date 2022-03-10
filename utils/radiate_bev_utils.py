import numpy as np
import math
import torch
import cv2
import utils.config as cnf

def read_labels_for_bevbox(objects):
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.w, obj.h, obj.ry])
            bbox_selected.append(bbox)
    if (len(bbox_selected) == 0):
        return np.zeros((1, 6), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False

# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
                      
    bev_corners = np.array([[x, y],
                        [x + w, y],
                        [x + w, y + l],
                        [x, y + l]]).T

    cx = x + w / 2
    cy = y + l / 2
    T = np.array([[cx], [cy]])

    bev_corners = bev_corners - T
    bev_corners = np.matmul(R, bev_corners) + T
    #corners = corners.astype(int)
    """
    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)
    """
    return bev_corners.T

def build_yolo_target(labels):
    target = np.zeros([50, 7], dtype=np.float32)
    index = 0
    for i in range(labels.shape[0]):
        cl, x, y, w, h, yaw = labels[i]
        # ped and cyc labels are very small, so lets add some factor to height/width
        h = h + 0.3
        w = w + 0.3

        yaw = np.pi * 2 - yaw
        y1 = y / cnf.BEV_WIDTH
        x1 = x / cnf.BEV_HEIGHT
        h1 = h / cnf.BEV_WIDTH
        w1 = w / cnf.BEV_HEIGHT

        target[index][0] = cl
        target[index][1] = y1 
        target[index][2] = x1
        target[index][3] = h1
        target[index][4] = w1
        target[index][5] = math.sin(float(yaw))
        target[index][6] = math.cos(float(yaw))

        index = index+1
    return target

#send parameters in bev image coordinates format
def drawRotatedBox(img,x,y,w,l,yaw,color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)

def draw_box_in_bev(rgb_map, target):
    for j in range(50):
        if(np.sum(target[j,1:]) == 0):continue
        cls_id = int(target[j][0])
        x = target[j][1] * cnf.BEV_WIDTH
        y = target[j][2] * cnf.BEV_HEIGHT
        w = target[j][3] * cnf.BEV_WIDTH
        l = target[j][4] * cnf.BEV_HEIGHT
        yaw = np.arctan2(target[j][5], target[j][6])
        drawRotatedBox(rgb_map, x, y, w, l, yaw, cnf.colors[cls_id])
