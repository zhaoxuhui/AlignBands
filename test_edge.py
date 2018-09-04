# coding=utf-8
import numpy as np
import cv2


def nothing(x):
    pass


img = cv2.imread("E:\\edge\\img2.PNG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

minVal = 100
maxVal = 200
switch = 1
result = gray

cv2.namedWindow("Canny Test")

cv2.createTrackbar('minVal', 'Canny Test', minVal, 255, nothing)
cv2.createTrackbar('maxVal', 'Canny Test', maxVal, 255, nothing)
cv2.createTrackbar('0:Off 1:On', 'Canny Test', switch, 1, nothing)

while 1:
    cv2.imshow('Canny Test', result)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    minVal = cv2.getTrackbarPos('minVal', 'Canny Test')
    maxVal = cv2.getTrackbarPos('maxVal', 'Canny Test')
    switch = cv2.getTrackbarPos('0:Off 1:On', 'Canny Test')

    if switch == 1:
        # 第一个参数是待处理的图像
        # 第二个参数是最小阈值minVal
        # 第三个参数是最大阈值maxVal
        # 第四个参数是Sobel算子卷积核大小，默认为3，可省略
        # 第五个参数是L2Gradient，用于设定求梯度大小的方程。如果为True，则用开根号的那个，否则用绝对值的那个，默认为False，可省略
        result = cv2.Canny(gray, minVal, maxVal)
    else:
        result = gray

cv2.destroyAllWindows()
