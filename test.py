# coding=utf-8
import cv2
import numpy as np
import functions as fun
from matplotlib import pyplot as plt

img = fun.loadIMGintoMem("/root/004/HAZ1_20180719222834_0004_L1_MSS_B14_CCD2.tif")
plt.imshow(img, cmap='gray')
plt.show()

# 云过滤
# band_b_10 = fun.loadIMGintoMem(
#     "C:\\Users\\obtdata005\\Desktop\\2632\\test\\global\\HDZ2_20180712222632_0007_L0_MSS_RC_B03_CCD3.tif")
# band_b, old_min_b, old_max_b = fun.linearStretchWithData(
#     np.uint8(np.where(band_b_10 == 0, 128, band_b_10) / 4), 1, 255, 0.02)
# surf = cv2.xfeatures2d_SURF.create(hessianThreshold=1200)
# kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, band_b, None)
# good_kps, good_res, removed_kps, removed_des, mask, img_out = fun.cloudFilter(band_b, kp, des, ksize=7, iter=6)
# cv2.imshow("out", img_out)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\\HAZ2_20180713222912\\cut\\\png\\b02_kps.jpg", img_out)
# cv2.waitKey(0)

# # 云过滤匹配效果
# img_base = cv2.imread("C:\\Users\\obtdata005\\Desktop\\\HAZ2_20180713222912\\cut\\\png\\b02.png", cv2.IMREAD_GRAYSCALE)
# img_resample = cv2.imread("C:\\Users\\obtdata005\\Desktop\\\HAZ2_20180713222912\\cut\\\png\\b07.png",
#                           cv2.IMREAD_GRAYSCALE)
# kps1, kps2, matches, img = fun.FLANN_SURF_AutoTh_Cloud(img_base, img_resample, 1500, 250, 100, 30000)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\\HAZ2_20180713222912\\cut\\\png\\match.jpg", img)

# # 不同算子提取特征匹配对比
# img_base = cv2.imread("C:\\Users\\obtdata005\\Desktop\\DemoData\\band04.png", cv2.IMREAD_GRAYSCALE)
# img_resample = cv2.imread("C:\\Users\\obtdata005\\Desktop\\DemoData\\band22.png", cv2.IMREAD_GRAYSCALE)
#
# kps1, kps2, matches, img = fun.FLANN_SURF(img_base, img_resample, 1500)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\match_surf.jpg", img)
#
# kps3, kps4, matches2, img2 = fun.BF_ORB(img_base, img_resample, 3000)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\match_orb.jpg", img2)
#
# kps5, kps6, matches3, img3 = fun.FLANN_SIFT(img_base, img_resample, 10000)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\match_sift.jpg", img3)
#
# good_matches = []
# good_matches.extend(matches)
# good_matches.extend(matches2)
# good_matches.extend(matches3)
# print(good_matches.__len__())
# out = fun.drawMatches(img_base, img_resample, good_matches)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\join.jpg", out)


# # 验证预处理效果
# img = fun.loadIMGintoMem("C:\\Users\\obtdata005\\Desktop\\DemoData\\out2.tif")
# img_8bit = np.uint8(img / 4)
# img_8bit_str, old_min_b, old_max_b = fun.linearStretchWithData(np.uint8(np.where(img == 0, 128, img) / 4), 1, 255, 0.02)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\surf8.jpg", img_8bit)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\surf8_str.jpg", img_8bit_str)
# surf = cv2.xfeatures2d_SURF.create(hessianThreshold=2000)
# kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img_8bit, None)
# kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img_8bit_str, None)
# print(len(kp1))
# print(len(kp2))
# print(img_8bit.shape.__len__())
# img_8bit_kp = fun.drawKeyPoints(img_8bit, kp1)
# img_8bit_str_kp = fun.drawKeyPoints(img_8bit_str, kp2)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\surf8_kp.jpg", img_8bit_kp)
# cv2.imwrite("C:\\Users\\obtdata005\\Desktop\\DemoData\\surf8_str_kp.jpg", img_8bit_str_kp)
