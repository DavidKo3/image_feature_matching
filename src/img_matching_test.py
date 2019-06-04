import cv2
import numpy as np
from scipy.linalg import logm, expm
import math
# template_path = "../data/deal-tshirt/"
# template_filename = "deal-tshirt-1-2.png"
#
# sample_path = "../data/deal-tshirt/"
# sample_fiilename = "deal-tshirt.jpg"

#
# template_path = "../data/scrapy_data/neat/"
# template_filename = "[5컬러_니쥬_브이넥_골지_루즈핏_니트]_0_.jpg"
#
# sample_path = "../data/scrapy_data/neat/"
# sample_fiilename = "[5컬러_니쥬_브이넥_골지_루즈핏_니트]_1_.jpg"


template_path = "../data/scrapy_data/neat_crop/"
template_filename = "neat_0.png"

sample_path = "../data/scrapy_data/neat_crop/"
sample_fiilename = "neat_4.png"



source_img = cv2.imread(template_path + template_filename, 0)
target_img = cv2.imread(sample_path + sample_fiilename, 0)


(w, h) = source_img.shape
(t_w, t_h) = target_img.shape

# print(w,h )
# print(t_w, t_h)
target_img = cv2.resize(target_img.copy(), (h,w), interpolation=cv2.INTER_CUBIC)
# print(target_img.shape)
def log_euclidean_matrix(feature_mat):
    (U, s, V) = np.linalg.svd(feature_mat, full_matrices = True)
    print(U.shape)
    print(s.shape)
    print(V.shape)
    # svd = np.dot(U, np.diag(s)).dot(V)
    S= np.zeros(feature_mat.shape)
    for i in range(len(s)):
        S[i][i]=s[i]

    log_s = np.zeros(feature_mat.shape)
    for i in range(len(s)):
        log_s[i][i]=math.log(S[i][i])

    log_svd = np.dot(U, np.dot(log_s,V))
    return log_svd


log_euc_source = log_euclidean_matrix(source_img)
log_euc_target = log_euclidean_matrix(target_img)

diff_source_target = log_euc_source-log_euc_target


avg_diff = abs(diff_source_target.sum()/diff_source_target.shape[0])


print(avg_diff)
# print(log_svd)
# print(template_temp-svd)
# result_path = ""
# result_name = ""
#
# akaze = cv2.AKAZE_create()
#
#
# # Read chracter image and calculate feature quantity
# expand_template = 2
# whitespace = 20
#
# template_temp = cv2.imread(template_path + template_filename, 0)
# height, width = template_temp.shape[:2]
# template_img = np.ones((height + whitespace*2, width + whitespace*2), np.uint8) * 255
#
# template_img[whitespace:whitespace+ height, whitespace:whitespace+width] = template_temp
# template_img = cv2.resize(template_img, None, fx = expand_template, fy = expand_template)
# kp_temp, des_temp = akaze.detectAndCompute(template_img, None)
#
# # Load floor plan and calculate feature quantity
# expand_sample = 2
# sample_img = cv2.imread(sample_path+ sample_fiilename, 0)
# sample_img = cv2.resize(sample_img, None, fx=expand_sample, fy=expand_sample)
# kp_samp, des_samp = akaze.detectAndCompute(sample_img, None)
#
#
# # Feature amount matching execution
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des_temp, des_samp, k=2)
#
# # Extract only those with high matching accuracy
# ratio = 0.5
# good = []
# for m, n in matches:
#     if m.distance < ratio*n.distance:
#         good.append([m])
#
#
# # Draw and save the matching result
# cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO|cv2.WINDOW_NORMAL)
# result_img = cv2.drawMatchesKnn(template_img, kp_temp, sample_img, kp_samp, good, None, flags=0)
# cv2.imshow("Result", result_img)
# cv2.waitKey(0)














