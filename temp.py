from __future__ import print_function
import numpy as np
import cv2

import pandas as pd
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2 as cv2
from itertools import chain
from scipy.interpolate import interp1d
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
# import os
#
#
# path = 'experiments/failed exps/190520T2pro/'
# # img = cv2.imread(f'{path}{files[0]}')
# count_files = len(os.listdir(path))
# print(count_files)
#
# for i in range(1, 11, 2):
#     print(i)




MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.30

img1 = cv2.imread('experiments/failed exps/190520T2pro/190520T2pro00000000.jpg')
img2 = cv2.imread('experiments/failed exps/190520T2pro/190520T2pro00000499.jpg')

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    print(numGoodMatches)
    plt.imshow(imMatches)
    plt.show()
    cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # print(type(h), type(mask))
    # print()
    # print(h)
    # print()
    # print(mask)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h

img1_mod, h = alignImages(img1, img2)

plt.imshow(img1)
plt.imshow(img1_mod)
plt.show()

print(h)

# if __name__ == '__main__':
#     # Read reference image
#     refFilename = "form.jpg"
#     print("Reading reference image : ", refFilename)
#     imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#     # Read image to be aligned
#     imFilename = "scanned-form.jpg"
#     print("Reading image to align : ", imFilename);
#     im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
#     print("Aligning images ...")
#     # Registered image will be resotred in imReg.
#     # The estimated homography will be stored in h.
#     imReg, h = alignImages(im, imReference)
#     # Write aligned image to disk.
#     outFilename = "aligned.jpg"
#     print("Saving aligned image : ", outFilename);
#     cv2.imwrite(outFilename, imReg)
#     # Print estimated homography
#     print("Estimated homography : \n", h)

# img=cv.drawKeypoints(gray,kp,img)

# cv.imshow("SIFT", img)
# cv.imwrite('sift_keypoints.jpg',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
