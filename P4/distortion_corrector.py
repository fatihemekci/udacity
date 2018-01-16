import cv2
import glob
import numpy as np
import time

import PyQt4
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

class DistortionCorrector:

    def __init__(self):
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.dist = None
        self.mtx = None

    def train(self, file_names="", nx=8, ny=6):
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        images = glob.glob(file_names)
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                # write_name = 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(write_name, img)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

        # cv2.destroyAllWindows()

    def cal_undistort(self, img, color_schema=cv2.COLOR_BGR2GRAY, verbose=0):
        gray_img = cv2.cvtColor(img, color_schema)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray_img.shape[::], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        if verbose != 0:
            cv2.imshow('img', undist)
            cv2.waitKey(500)
            print("plotting")
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(undist)
            ax2.set_title('Undistorted Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            time.sleep(20)
        return undist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
