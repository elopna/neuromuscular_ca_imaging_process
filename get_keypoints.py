import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = 'experiments/failed exps/190520T2pro/190520T2pro00000000.jpg'

img1 = cv2.imread(filename)          # queryImage
# img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB()

plt.imshow(img1)
# plt.show()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# print(kp1, des1)