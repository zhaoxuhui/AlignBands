# coding=utf-8
import cv2
import numpy as np
import functions as fun
from matplotlib import pyplot as plt

# band_1 = cv2.imread("E:\\Desktop\\img1.png", cv2.IMREAD_GRAYSCALE)
# band_2 = cv2.imread("E:\\Desktop\\img2.png", cv2.IMREAD_GRAYSCALE)
# band_3 = cv2.imread("E:\\Desktop\\img3.png", cv2.IMREAD_GRAYSCALE)
#
# ed1 = cv2.Canny(band_1, 10, 20)
# ed2 = cv2.Canny(band_2, 10, 20)
# ed3 = cv2.Canny(band_3, 10, 20)
# cv2.imshow("ed1", ed1)
# cv2.imshow("ed2", ed2)
# cv2.imshow("ed3", ed3)
# cv2.waitKey(0)
# kps1, kps2, matches, img = fun.FLANN_SURF(ed1, ed3, 1500)
# cv2.imwrite("E:\\Desktop\\ed_match_surf13.jpg", img)
# kps1, kps2, matches, img = fun.FLANN_SURF(ed1, ed2, 1500)
# cv2.imwrite("E:\\Desktop\\ed_match_surf12.jpg", img)

band1 = cv2.imread("E:\\Desktop\\img1.png", cv2.IMREAD_GRAYSCALE)
band2 = cv2.imread("E:\\Desktop\\img2.png", cv2.IMREAD_GRAYSCALE)
# ed1 = cv2.Canny(band1, 0, 60)
# ed2 = cv2.Canny(band2, 0, 60)
# cv2.imshow("ed1", ed1)
# cv2.imshow("ed2", ed2)
# cv2.waitKey(0)
kps1, kps2, matches, img = fun.FLANN_SURF(band1, band2, 1000)
cv2.imwrite("E:\\Desktop\\ed_match_surf12.jpg", img)

# kps1, kps2, matches, img = fun.FLANN_SURF(band_1, band_3, 1500)
# cv2.imwrite("E:\\Desktop\\match_surf13.jpg", img)
# kps12, kps22, matches2, img2 = fun.FLANN_SIFT(band_1, band_3, 10)
# cv2.imwrite("E:\\Desktop\\match_sift13.jpg", img2)
# kps13, kps23, matches3, img3 = fun.BF_ORB(band_1, band_3, 10)
# cv2.imwrite("E:\\Desktop\\match_orb13.jpg", img3)
#
# kps1, kps2, matches, img = fun.FLANN_SURF(band_1, band_2, 1500)
# cv2.imwrite("E:\\Desktop\\match_surf12.jpg", img)
# kps12, kps22, matches2, img2 = fun.FLANN_SIFT(band_1, band_2, 10)
# cv2.imwrite("E:\\Desktop\\match_sift12.jpg", img2)
# kps13, kps23, matches3, img3 = fun.BF_ORB(band_1, band_2, 10)
# cv2.imwrite("E:\\Desktop\\match_orb12.jpg", img3)
