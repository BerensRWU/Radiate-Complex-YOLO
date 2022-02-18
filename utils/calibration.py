import numpy as np

class Calibration:
    """
    Calibration Class for the Radiate Dataset
    """
    def __init__(self, cfg):
        # extrinsic
        self.radarT = np.array(cfg['radar_calib']['T'])
        self.radarR = np.array(cfg['radar_calib']['R'])
        self.lidarT = np.array(cfg['lidar_calib']['T'])
        self.lidarR = np.array(cfg['lidar_calib']['R'])

        self.radar2lidarT = self.radarT - self.lidarT
        self.radar2lidarR = self.radarR - self.lidarR
        self.radar2lidar = self.transform(self.radar2lidarR, self.radar2lidarT)

        self.lidar2radar = self.transform(self.lidarR, self.lidarT)
    
    def RX(self, eulervector):
        thetaX = np.deg2rad(eulervector[0])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(thetaX), -np.sin(thetaX)],
                       [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float)
        return Rx

    def RY(self, eulervector):
        thetaY = np.deg2rad(eulervector[1])
        Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                       [0, 1, 0],
                       [-np.sin(thetaY), 0, np.cos(thetaY)]])
        return Ry

    def RZ(self, eulervector):
        thetaZ = np.deg2rad(eulervector[2])
        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                       [np.sin(thetaZ), np.cos(thetaZ), 0],
                       [0, 0, 1]]).astype(np.float)
        return Rz

    def transform(self, eulervector, transvector):
        Rx = self.RX(eulervector)
        Ry = self.RY(eulervector)
        Rz = self.RZ(eulervector)

        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]]).astype(np.float)
        R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

        transformMatrix = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                               [R[1, 0], R[1, 1], R[1, 2], 0.0],
                               [R[2, 0], R[2, 1], R[2, 2], 0.0],
                               [transvector[0], transvector[1], transvector[2], 1.0]]).T
        return transformMatrix
