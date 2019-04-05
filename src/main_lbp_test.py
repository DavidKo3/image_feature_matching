import numpy as np
import cv2
from matplotlib import pyplot as plt
from data.pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from imutils import paths
import argparse
import glob


def draw_bbox(target_img, target_keypts, matched_max, option):
    results=[]

    if option=='orb':
        for match in matched_max:
            ind = match.trainIdx
            point = target_keypts[ind].pt
            results.append((int(point[0]), int(point[1])))

        min_x, min_y = np.min(results, axis=0)
        max_x, max_y = np.max(results, axis=0)

        w, h = int(max_x - min_x), int(max_y - min_y)
        cv2.rectangle(target_img, (min_x, min_y), (min_x + w, min_y + h), (0, 255, 0), 2)

        return target_img
    else:
        for match in matched_max:
            ind = match[0].trainIdx
            point = target_keypts[ind].pt
            results.append((int(point[0]), int(point[1])))

        min_x, min_y = np.min(results, axis=0)
        max_x, max_y = np.max(results, axis=0)

        w, h = int(max_x - min_x), int(max_y - min_y)
        cv2.rectangle(target_img, (min_x, min_y), (min_x + w, min_y + h), (0, 255, 0), 2)

        return target_img



th = 10

# img1 = cv2.imread("./data/deal-tshirt/deal-tshirt-1-3.png", 0)    #query image
# img2 = cv2.imread("./data/deal-tshirt/deal-tshirt.jpg")  # train image
# target_origin = img2.copy()
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#
# fig = plt.figure(figsize=(30, 20))
#
# radius = 3
# # Number of points to be considered as neighbourers
# no_points = 8 * radius
#
#
# img1_lbp = LocalBinaryPatters(no_points, radius).describe(img1)
# img2_lbp = LocalBinaryPatters(no_points, radius).describe(img2)
#
# score = cv2.compareHist(np.array(img1_lbp, dtype=np.float32), np.array(img2_lbp, dtype=np.float32), cv2.HISTCMP_CHISQR)
# print(score)


radius = 3
# Number of points to be considered as neighbourers
no_points = 8 * radius

test_list = sorted(glob.glob("../data/denim/*.*"))


# img1 = cv2.imread("./data/deal-tshirt/deal-tshirt-1-3.png", 0)
img2 = cv2.imread("../data/deal-tshirt/deal-tshirt-1.png")  # train image
target_origin = img2.copy()
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


score = dict()
for test_img in test_list:
    img1= cv2.imread(test_img, 0) #query image
    img1_lbp = LocalBinaryPatterns(no_points, radius).describe(img1)
    img2_lbp = LocalBinaryPatterns(no_points, radius).describe(img2)

    val = cv2.compareHist(np.array(img1_lbp, dtype=np.float32), np.array(img2_lbp, dtype=np.float32),
                            cv2.HISTCMP_CHISQR)

    score[test_img.split('/')[3]]= val


temp = sorted(score.items(), key = lambda x:x[1])
for i in temp:
    print(i)


fig = plt.figure(figsize=(30, 20))









# orb = cv2.ORB()
# orb = cv2.ORB_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
#
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1, des2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:th], None, flags=2)
#
#
# ax1 = fig.add_subplot(2, 3, 1)
# ax1_1 = fig.add_subplot(2, 3, 4)
#
# ax1.imshow(img3)
# target_1 = target_origin.copy()
# ret_1 = draw_bbox(target_1, kp2, matches[:th], option='orb')
# ax1_1.imshow(ret_1)
#
#
#
# # Initiate SIFT detector
# # sift = cv2.SIFT()
# sift = cv2.xfeatures2d.SIFT_create()
#
#
# # find the keypoints and descriptors with SIFT
# kp3, des3 = sift.detectAndCompute(img1, None)
# kp4, des4 = sift.detectAndCompute(img2, None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des3,des4, k=2)
#
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
# img5 = cv2.drawMatchesKnn(img1,kp3,img2,kp4, good[:th], None, flags=2)
#
# ax2 = fig.add_subplot(2, 3, 2)
# ax2_1 = fig.add_subplot(2, 3, 5)
#
# ax2.imshow(img5)
# target_2 = target_origin.copy()
# ret_2 = draw_bbox(target_2, kp4, good[:th], option='sift')
# ax2_1.imshow(ret_2)
#
#
#
# # Initiate SURF detector
#
# # surf = cv2.xfeatures2d.SURF_create(400, 5, 5)
# # print(surf)
# surf = cv2.xfeatures2d.SURF_create()
#
# # find the keypoints and descriptors with SIFT
# kp5, des5 = surf.detectAndCompute(img1, None)
# kp6, des6 = surf.detectAndCompute(img2, None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des5, des6, k=2)
#
# # Apply ratio test
# good = []
# for m, n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
#
# results =[]
# img6 = cv2.drawMatchesKnn(img1, kp5, img2, kp6, good[:th], None, flags=2)
#
#
# ax3 = fig.add_subplot(2, 3, 3)
# ax3_1 = fig.add_subplot(2, 3, 6)
#
# ax3.imshow(img6)
#
#
# target_3 = target_origin.copy()
# ret = draw_bbox(target_3, kp6, good[:th], option='surf')
# ax3_1.imshow(ret)
#
# plt.show()
#
# cv2.imshow("result", ret)
# cv2.waitKey(0)
# cv2.destroyAllWindows()