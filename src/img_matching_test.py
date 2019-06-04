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

print(w,h )
# print(t_w, t_h)
target_img = cv2.resize(target_img.copy(), (h,w), interpolation=cv2.INTER_CUBIC)
print(target_img.shape)
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

print("avg_diff :", avg_diff)

cv2.imshow("Result", target_img)
cv2.waitKey(0)














