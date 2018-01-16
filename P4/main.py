from distortion_corrector import DistortionCorrector

import cv2

def run():
    print('hello')
    distortion_corrector = DistortionCorrector()
    distortion_corrector.train(file_names="./camera_cal/cali*", nx=9, ny=6)
    img = cv2.imread("./camera_cal/calibration5.jpg")
    undis = distortion_corrector.cal_undistort(img, verbose=1)

if __name__ == '__main__':
    run()
