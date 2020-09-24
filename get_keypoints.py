import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

filename = 'experiments/failed exps/190520T2pro/190520T2pro00000000.jpg'
path = 'experiments/failed exps/190520T2pro/'

files = os.listdir(path)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.30

img1 = cv2.imread('experiments/failed exps/190520T2pro/190520T2pro00000000.jpg')
img2 = cv2.imread('experiments/failed exps/190520T2pro/190520T2pro00000499.jpg')

def alignImages(im1, im2, path1):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.30
    im1 = cv2.imread(f'{path1}{im1}')
    im2 = cv2.imread(im2)
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
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # print(numGoodMatches)
    # plt.imshow(imMatches)
    # plt.show()
    # cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return h, width, height

# h, width, height = alignImages(img1, img2)


# im1Reg = cv2.warpPerspective(im1, h, (width, height))








# for file in files[:10]:
#     img = cv2.imread(filename)          # queryImage
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     detector = cv2.xfeatures2d_SURF.create()
#     keypoints = detector.detect(img)
#     img=cv2.drawKeypoints(img,keypoints,img)
#     plt.imshow(img)
#     plt.show()






# img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
# orb = cv2.ORB()

# plt.imshow(img)
# plt.show()


# gray = cv2.cvtColor(img,cv.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv2.drawKeypoints(gray,kp,img)
#
# cv2.imshow("SIFT", img)
# cv2.imwrite('sift_keypoints.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# print(kp1, des1)