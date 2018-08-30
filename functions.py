# coding=utf-8
import os
import cv2
import numpy as np
from osgeo import gdal
from gdalconst import *
import time

# 配置参数默认值
win_w = 2000
win_h = 5000
stripe_height = 500
stripe_extension = 250
affine_min = 0.98
affine_max = 1.02
surf_th = 1500
surf_th_d = 500
sift_th = 10000
orb_th = 2000
stretch_th = 80
kps_max = 20000
kps_min = 500
isDebugMode = 1
isReverse = 0
block_overlap = 50
surf_th_global = 1500
surf_th_d_global = 500
sift_th_global = 10000
orb_th_global = 2000
kps_max_global = 20000
kps_min_global = 500
cloud_iter_num = 6
cloud_ksize = 7
isIteration = 1
isCloudMode = 0
cloud_th = -1
processNum = 6

global_counter = 0


def readConfigFile(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)

    # 申明为全局变量，否则赋值会失败
    global win_w
    global win_h
    global stripe_height
    global stripe_extension
    global affine_min
    global affine_max
    global surf_th
    global surf_th_d
    global sift_th
    global orb_th
    global stretch_th
    global kps_max
    global kps_min
    global isDebugMode
    global isReverse
    global block_overlap
    global surf_th_global
    global surf_th_d_global
    global sift_th_global
    global orb_th_global
    global kps_max_global
    global kps_min_global
    global cloud_iter_num
    global cloud_ksize
    global isIteration
    global isCloudMode
    global cloud_th
    global processNum

    win_w = int(fs.getNode("win_w").real())
    win_h = int(fs.getNode('win_h').real())
    stripe_height = int(fs.getNode('stripe_height').real())
    stripe_extension = int(fs.getNode('stripe_extension').real())
    block_overlap = int(fs.getNode('block_overlap').real())
    affine_min = fs.getNode('affine_min').real()
    affine_max = fs.getNode('affine_max').real()
    surf_th = int(fs.getNode('surf_th').real())
    surf_th_d = int(fs.getNode('surf_th_d').real())
    sift_th = int(fs.getNode('sift_th').real())
    orb_th = int(fs.getNode('orb_th').real())
    stretch_th = int(fs.getNode('stretch_th').real())
    kps_max = int(fs.getNode('kps_max').real())
    kps_min = int(fs.getNode('kps_min').real())
    isDebugMode = int(fs.getNode('isDebugMode').real())
    isReverse = int(fs.getNode('isReverse').real())
    surf_th_global = int(fs.getNode('surf_th_global').real())
    surf_th_d_global = int(fs.getNode('surf_th_d_global').real())
    sift_th_global = int(fs.getNode('sift_th_global').real())
    orb_th_global = int(fs.getNode('orb_th_global').real())
    kps_max_global = int(fs.getNode('kps_max_global').real())
    kps_min_global = int(fs.getNode('kps_min_global').real())
    cloud_iter_num = int(fs.getNode('cloud_iter_num').real())
    cloud_ksize = int(fs.getNode('cloud_ksize').real())
    isIteration = int(fs.getNode('isIteration').real())
    isCloudMode = int(fs.getNode('isCloudMode').real())
    cloud_th = int(fs.getNode('cloud_th').real())
    processNum = int(fs.getNode('processNum').real())

    if isDebugMode == 0:
        isDebugMode = False
    else:
        isDebugMode = True

    if isReverse == 0:
        isReverse = False
    else:
        isReverse = True

    if isIteration == 0:
        isIteration = False
    else:
        isIteration = True

    if isCloudMode == 0:
        isCloudMode = False
    else:
        isCloudMode = True

    print('=>config file read success')


def findAllFiles(root_dir, filter):
    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    # 遍历
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def findAllFilesReverse(root_dir, filter):
    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    # 遍历
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    paths.reverse()
    names.reverse()
    files.reverse()
    return paths, names, files


def drawKeyPoints(img, kps, color=(0, 0, 255), rad=3):
    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img
    for point in kps:
        pt = (int(point.pt[0]), int(point.pt[1]))
        cv2.circle(img_pro, pt, rad, color, 1, cv2.LINE_AA)
    return img_pro


def drawMatches(img1, img2, good_matches):
    if img1.shape.__len__() == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape.__len__() == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
    img_out[:img1.shape[0], :img1.shape[1], :] = img1
    img_out[:img2.shape[0], img1.shape[1]:, :] = img2
    for match in good_matches:
        pt1 = (int(match[0]), int(match[1]))
        pt2 = (int(match[2] + img1.shape[1]), int(match[3]))
        cv2.circle(img_out, pt1, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img_out, pt2, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return img_out


def generateOutputFilename(output_path, filenames, filetype):
    out_filenames = []
    for i in range(filenames.__len__()):
        if i == 0:
            continue
        temp = filenames[i].split('.')[0]
        temp = output_path + os.path.sep + temp + "_out." + filetype
        out_filenames.append(temp)
    return out_filenames


def loadIMGintoMem(img_path):
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    band_1 = dataset.GetRasterBand(1)
    data = band_1.ReadAsArray(0, 0, band_1.XSize, band_1.YSize)
    return data


def hist_calc(img, ratio):
    bins = np.arange(256)
    hist, bins = np.histogram(img, bins)
    total_pixels = img.shape[0] * img.shape[1]
    min_index = int(ratio * total_pixels)
    max_index = int((1 - ratio) * total_pixels)
    min_gray = 0
    max_gray = 0
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > min_index:
            min_gray = i
            break
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > max_index:
            max_gray = i
            break
    return min_gray, max_gray


def linearStretch(img, new_min, new_max, ratio):
    old_min, old_max = hist_calc(img, ratio)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    print("=>2% linear stretch:")
    print('old min = %d,old max = %d new min = %d,new max = %d' % (old_min, old_max, new_min, new_max))
    img3 = np.uint8((new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min)
    return img3


def linearStretchWithData(img, new_min, new_max, ratio):
    old_min, old_max = hist_calc(img, ratio)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    print("=>2% linear stretch:")
    print('old min = %d,old max = %d new min = %d,new max = %d' % (old_min, old_max, new_min, new_max))
    img3 = np.uint8((new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min)
    return img3, old_min, old_max


def SURF_Keypoints(img1, img2, threshold):
    # 新建SURF对象，参数默认
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold)
    # 调用函数进行SURF提取
    kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
    kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
    return kp1.__len__(), kp2.__len__()


def getProperSURFKps(img, f_surf_th, f_surf_th_d, f_kps_min, f_kps_max):
    curTh = f_surf_th
    # 新建SURF对象，参数默认
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    # 如果数量大于最大值，迭代增加阈值
    if kp.__len__() > f_kps_max:
        print(kp.__len__().__str__() + ' is more than ' + f_kps_max.__str__() + ' try to use larger th.')
        while kp.__len__() > f_kps_max:
            curTh = curTh + f_surf_th_d
            surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
            kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
            print('current th:' + curTh.__str__() + ' current kps:' + kp.__len__().__str__())
        return kp, des, curTh
    # 如果数量小于最小值，迭代减少阈值，最小为0
    elif kp.__len__() < f_kps_min:
        print(kp.__len__().__str__() + ' is less than ' + f_kps_min.__str__() + ' try to use smaller th.')
        while kp.__len__() < f_kps_min:
            curTh = curTh - f_surf_th_d
            # 如果出现curTh小于0的情况，直接将curTh赋为0，以此为条件进行检测，最后返回
            if curTh <= 0:
                curTh = 0
                surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
                kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
                print('current th:' + curTh.__str__() + ' current kps:' + kp.__len__().__str__())
                return kp, des, curTh
            else:
                surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
                kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
                print('current th:' + curTh.__str__() + ' current kps:' + kp.__len__().__str__())
        return kp, des, curTh
    # 数量在正常范围内，直接返回提取结果
    else:
        return kp, des, curTh


def getProperSURFKpsCloud(img, f_surf_th, f_surf_th_d, f_kps_min, f_kps_max):
    curTh = f_surf_th
    # 新建SURF对象
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    good_kps, good_res, removed_kps, removed_des, mask, img_out = cloudFilter(img, kp, des,
                                                                              ksize=cloud_ksize,
                                                                              iter=cloud_iter_num)

    # 如果数量大于最大值，迭代增加阈值
    if good_kps.__len__() > f_kps_max:
        print(good_kps.__len__().__str__() + ' is more than ' + f_kps_max.__str__() + ' try to use larger th.')
        while good_kps.__len__() > f_kps_max:
            curTh = curTh + f_surf_th_d
            surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
            kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
            good_kps, good_res, removed_kps, removed_des, mask, img_out = cloudFilter(img, kp, des,
                                                                                      ksize=cloud_ksize,
                                                                                      iter=cloud_iter_num)
            print('current th:' + curTh.__str__() + ' current kps:' + kp.__len__().__str__())
        return good_kps, good_res, curTh
    # 如果数量小于最小值，迭代减少阈值，最小为0
    elif good_kps.__len__() < f_kps_min:
        print(good_kps.__len__().__str__() + ' is less than ' + f_kps_min.__str__() + ' try to use smaller th.')
        while good_kps.__len__() < f_kps_min:
            curTh = curTh - f_surf_th_d
            # 如果出现curTh小于0的情况，直接将curTh赋为0，以此为条件进行检测，最后返回
            if curTh <= 0:
                curTh = 0
                surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
                kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
                good_kps, good_res, removed_kps, removed_des, mask, img_out = cloudFilter(img, kp, des,
                                                                                          ksize=cloud_ksize,
                                                                                          iter=cloud_iter_num)
                print('current th:' + curTh.__str__() + ' current kps:' + good_kps.__len__().__str__())
                return good_kps, good_res, curTh
            else:
                surf = cv2.xfeatures2d_SURF.create(hessianThreshold=curTh)
                kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
                good_kps, good_res, removed_kps, removed_des, mask, img_out = cloudFilter(img, kp, des,
                                                                                          ksize=cloud_ksize,
                                                                                          iter=cloud_iter_num)
                print('current th:' + curTh.__str__() + ' current kps:' + good_kps.__len__().__str__())
        return good_kps, good_res, curTh
    # 数量在正常范围内，直接返回提取结果
    else:
        return good_kps, good_res, curTh


def FLANN_SURF_AutoTh_Cloud(img1, img2, fsurf_th, fsurf_th_d, fkps_min, fkps_max):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 获取合适数量的特征点
    kp1, des1, th1 = getProperSURFKpsCloud(img1, fsurf_th, fsurf_th_d, fkps_min, fkps_max)
    kp2, des2, th2 = getProperSURFKpsCloud(img2, fsurf_th, fsurf_th_d, fkps_min, fkps_max)

    # 如果特征点数量小于3，认为特征点数量太少，直接返回
    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    # 如果匹配结果为0，返回
    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def FLANN_SURF_AutoTh(img1, img2, fsurf_th, fsurf_th_d, fkps_min, fkps_max):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 获取合适数量的特征点
    kp1, des1, th1 = getProperSURFKps(img1, fsurf_th, fsurf_th_d, fkps_min, fkps_max)
    kp2, des2, th2 = getProperSURFKps(img2, fsurf_th, fsurf_th_d, fkps_min, fkps_max)

    # 如果特征点数量小于3，认为特征点数量太少，直接返回
    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    # 如果匹配结果为0，返回
    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def FLANN_SURF(img1, img2, threshold):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 新建SURF对象，参数默认
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold)
    # 调用函数进行SURF提取
    kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
    kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    elif len(kp1) > kps_max or len(kp2) > kps_max:
        print("Too many keypoints,use bigger threshold.")
        surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold + surf_th_d)
        kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
        kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())
        if len(kp1) < 3 or len(kp2) < 3:
            print("No enough keypoints, use former threshold.")
            surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold)
            kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
            kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
            print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def FLANN_SURFCloud(img1, img2, threshold):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 新建SURF对象，参数默认
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold)
    # 调用函数进行SURF提取
    kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
    kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
    kp1, des1, rm_kp1, rm_des1, mask1, img_new1 = cloudFilter(img1, kp1, des1)
    kp2, des2, rm_kp2, rm_des2, mask2, img_new2 = cloudFilter(img2, kp2, des2)

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    elif len(kp1) > kps_max or len(kp2) > kps_max:
        print("Too many keypoints,use bigger threshold.")
        surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold + surf_th_d)
        kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
        kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())
        if len(kp1) < 3 or len(kp2) < 3:
            print("No enough keypoints, use former threshold.")
            surf = cv2.xfeatures2d_SURF.create(hessianThreshold=threshold)
            kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
            kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)
            print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def FLANN_SIFT(img1, img2, fsift):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 新建SIFT对象，参数默认
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=fsift)
    # 调用函数进行SIFT提取
    kp1, des1 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img1, None)
    kp2, des2 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img2, None)

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:

        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def FLANN_SIFTCloud(img1, img2, fsift):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    # 新建SIFT对象，参数默认
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=fsift)
    # 调用函数进行SIFT提取
    kp1, des1 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img1, None)
    kp2, des2 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img2, None)
    kp1, des1, rm_kp1, rm_des1, mask1, img_new1 = cloudFilter(img1, kp1, des1)
    kp2, des2, rm_kp2, rm_des2, mask2, img_new2 = cloudFilter(img2, kp2, des2)

    if len(kp1) < 3 or len(kp2) < 3:
        print("No enough keypoints, keypoint number is less than 3.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_out_kp1, good_out_kp2, good_out, img_out
    else:

        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    return good_out_kp1, good_out_kp2, good_out, img3


def BF_ORB(img1, img2, forb_th):
    good_matches = []
    good_kps1 = []
    good_kps2 = []
    kp1 = []
    kp2 = []

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=forb_th)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if len(kp1) == 0 or len(kp2) == 0:
        print("No enough keypoints.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_kps1, good_kps2, good_matches, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    if matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_kps1, good_kps2, good_matches, img_out
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, 15.0):
                g_matches.append(match)

        print("matches:" + g_matches.__len__().__str__())

        # 筛选
        for i in range(g_matches.__len__()):
            good_kps1.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1]])
            good_kps2.append([kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])
            good_matches.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1],
                                 kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])

        # Draw matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, g_matches, None, flags=2)
        return good_kps1, good_kps2, good_matches, img3


def BF_ORBCloud(img1, img2, forb_th):
    good_matches = []
    good_kps1 = []
    good_kps2 = []
    kp1 = []
    kp2 = []

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=forb_th)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    kp1, des1, rm_kp1, rm_des1, mask1, img_new1 = cloudFilter(img1, kp1, des1)
    kp2, des2, rm_kp2, rm_des2, mask2, img_new2 = cloudFilter(img2, kp2, des2)

    if len(kp1) == 0 or len(kp2) == 0:
        print("No enough keypoints.")
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_out[:img2.shape[0], img1.shape[1]:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return good_kps1, good_kps2, good_matches, img_out
    else:
        print("kp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    if matches.__len__() == 0:
        print("No enough good matches.")
        img_show1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_show2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(img1, kp1, img_show1, color=(0, 0, 255))
        cv2.drawKeypoints(img2, kp2, img_show2, color=(0, 0, 255))
        img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
        img_out[:img1.shape[0], :img1.shape[1], :] = img_show1
        img_out[:img2.shape[0], img1.shape[1]:, :] = img_show2
        return good_kps1, good_kps2, good_matches, img_out
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, 15.0):
                g_matches.append(match)

        print("matches:" + g_matches.__len__().__str__())

        # 筛选
        for i in range(g_matches.__len__()):
            good_kps1.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1]])
            good_kps2.append([kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])
            good_matches.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1],
                                 kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])

        # Draw matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, g_matches, None, flags=2)
        return good_kps1, good_kps2, good_matches, img3


def getBandsOffsetWithNoStretch(img1, img2, number_counter):
    print('=>band offset detect')
    width1 = img1.shape[1]
    height1 = img1.shape[0]
    width2 = img2.shape[1]
    height2 = img2.shape[0]
    cen_w1 = width1 / 2
    cen_h1 = height1 / 2
    cen_w2 = width2 / 2
    cen_h2 = height2 / 2
    delta_x = 0
    delta_y = 0

    win1 = img1[cen_h1 - win_h / 2:cen_h1 + win_h / 2, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
    win2 = img2[cen_h2 - win_h / 2:cen_h2 + win_h / 2, cen_w2 - win_w / 2:cen_w2 + win_w / 2]
    kp1_size, kp2_size = SURF_Keypoints(win1, win2, threshold=surf_th)
    if max(kp1_size, kp2_size) < kps_min:
        print("Too little keypoints,use smaller threshold.")
        good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win1, win2, threshold=surf_th - surf_th_d)
    else:
        good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win1, win2, threshold=surf_th)

    if good_kp1.__len__() == 0:
        print("No good matches in center search window.Try top search window.")
        win3 = img1[:win_h, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
        win4 = img2[:win_h, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

        kp1_size, kp2_size = SURF_Keypoints(win3, win4, threshold=surf_th)
        if max(kp1_size, kp2_size) < kps_min:
            print("Too little keypoints,use smaller threshold.")
            good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win3, win4,
                                                                   threshold=surf_th - surf_th_d)
        else:
            good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win3, win4, threshold=surf_th)

        if good_kp1.__len__() == 0:
            print("No good matches in top search window.Try bottom search window.")
            win5 = img1[-win_h:, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
            win6 = img2[-win_h:, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

            kp1_size, kp2_size = SURF_Keypoints(win5, win6, threshold=surf_th)
            if max(kp1_size, kp2_size) < kps_min:
                print("Too little keypoints,use smaller threshold.")
                good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win5, win6, threshold=surf_th - surf_th_d)
            else:
                good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win5, win6, threshold=surf_th)
            if good_kp1.__len__() == 0:
                print("No good matches in center,top,bottom search window.Use 0 as default offset.")
                return delta_x, delta_y

    for i in range(good_kp1.__len__()):
        delta_x = delta_x + (good_kp1[i][0] - good_kp2[i][0])
        delta_y = delta_y + (good_kp1[i][1] - good_kp2[i][1])
    delta_x = delta_x / good_kp1.__len__()
    delta_y = delta_y / good_kp1.__len__()
    print('x offset:' + delta_x.__str__() + " y offset:" + delta_y.__str__())

    if isDebugMode:
        img_out = drawMatches(cv2.cvtColor(win1, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(win2, cv2.COLOR_GRAY2BGR),
                              good_matches)
        cv2.imwrite("output/win_match_" + number_counter.__str__() + ".jpg", img_out)
    return int(delta_x), int(delta_y)


def getBandsOffsetWithNoStretchWithData(img1, img2, number_counter):
    print('=>band offset detect')
    width1 = img1.shape[1]
    height1 = img1.shape[0]
    width2 = img2.shape[1]
    height2 = img2.shape[0]
    cen_w1 = width1 / 2
    cen_h1 = height1 / 2
    cen_w2 = width2 / 2
    cen_h2 = height2 / 2
    delta_x = 0
    delta_y = 0

    win1 = img1[cen_h1 - win_h / 2:cen_h1 + win_h / 2, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
    win2 = img2[cen_h2 - win_h / 2:cen_h2 + win_h / 2, cen_w2 - win_w / 2:cen_w2 + win_w / 2]
    kp1_size, kp2_size = SURF_Keypoints(win1, win2, threshold=surf_th)
    if max(kp1_size, kp2_size) < kps_min:
        print("Too little keypoints,use smaller threshold.")
        good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win1, win2, threshold=surf_th - surf_th_d)
    else:
        good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win1, win2, threshold=surf_th)

    if good_kp1.__len__() == 0:
        print("No good matches in center search window.Try top search window.")
        win3 = img1[:win_h, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
        win4 = img2[:win_h, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

        kp1_size, kp2_size = SURF_Keypoints(win3, win4, threshold=surf_th)
        if max(kp1_size, kp2_size) < kps_min:
            print("Too little keypoints,use smaller threshold.")
            good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win3, win4,
                                                                   threshold=surf_th - surf_th_d)
        else:
            good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win3, win4, threshold=surf_th)

        if good_kp1.__len__() == 0:
            print("No good matches in top search window.Try bottom search window.")
            win5 = img1[-win_h:, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
            win6 = img2[-win_h:, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

            kp1_size, kp2_size = SURF_Keypoints(win5, win6, threshold=surf_th)
            if max(kp1_size, kp2_size) < kps_min:
                print("Too little keypoints,use smaller threshold.")
                good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win5, win6, threshold=surf_th - surf_th_d)
            else:
                good_kp1, good_kp2, good_matches, img_out = FLANN_SURF(win5, win6, threshold=surf_th)
            if good_kp1.__len__() == 0:
                print("No good matches in center,top,bottom search window.Use 0 as default offset.")
                return delta_x, delta_y

    for i in range(good_kp1.__len__()):
        delta_x = delta_x + (good_kp1[i][0] - good_kp2[i][0])
        delta_y = delta_y + (good_kp1[i][1] - good_kp2[i][1])
    delta_x = delta_x / good_kp1.__len__()
    delta_y = delta_y / good_kp1.__len__()
    print('x offset:' + delta_x.__str__() + " y offset:" + delta_y.__str__())

    if isDebugMode:
        img_out = drawMatches(cv2.cvtColor(win1, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(win2, cv2.COLOR_GRAY2BGR),
                              good_matches)
        cv2.imwrite("output/win_match_" + number_counter.__str__() + ".jpg", img_out)
    return int(delta_x), int(delta_y), good_kp1, good_kp2


def getBandsOffsetWithNoStretchAuto(img1, img2, number_counter):
    print('=>band offset detect')
    width1 = img1.shape[1]
    height1 = img1.shape[0]
    width2 = img2.shape[1]
    height2 = img2.shape[0]
    cen_w1 = width1 / 2
    cen_h1 = height1 / 2
    cen_w2 = width2 / 2
    cen_h2 = height2 / 2
    delta_x = 0
    delta_y = 0

    win1 = img1[cen_h1 - win_h / 2:cen_h1 + win_h / 2, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
    win2 = img2[cen_h2 - win_h / 2:cen_h2 + win_h / 2, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

    good_kp1, good_kp2, good_matches, img_out = FLANN_SURF_AutoTh(win1, win2, surf_th, surf_th_d, kps_min, kps_max)
    if good_kp1.__len__() == 0:
        print("No good matches in center search window.Try top search window.")
        win3 = img1[:win_h, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
        win4 = img2[:win_h, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

        good_kp1, good_kp2, good_matches, img_out = FLANN_SURF_AutoTh(win3, win4, surf_th, surf_th_d, kps_min, kps_max)
        if good_kp1.__len__() == 0:
            print("No good matches in top search window.Try bottom search window.")
            win5 = img1[-win_h:, cen_w1 - win_w / 2:cen_w1 + win_w / 2]
            win6 = img2[-win_h:, cen_w2 - win_w / 2:cen_w2 + win_w / 2]

            good_kp1, good_kp2, good_matches, img_out = FLANN_SURF_AutoTh(win5, win6,
                                                                          surf_th,
                                                                          surf_th_d,
                                                                          kps_min,
                                                                          kps_max)
            if good_kp1.__len__() == 0:
                print("No good matches in center,top,bottom search window.Use 0 as default offset.")
                return delta_x, delta_y

    for i in range(good_kp1.__len__()):
        delta_x = delta_x + (good_kp1[i][0] - good_kp2[i][0])
        delta_y = delta_y + (good_kp1[i][1] - good_kp2[i][1])
    delta_x = delta_x / good_kp1.__len__()
    delta_y = delta_y / good_kp1.__len__()
    print('x offset:' + delta_x.__str__() + " y offset:" + delta_y.__str__())

    if isDebugMode:
        img_out = drawMatches(cv2.cvtColor(win1, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(win2, cv2.COLOR_GRAY2BGR),
                              good_matches)
        cv2.imwrite("output/win_match_" + number_counter.__str__() + ".jpg", img_out)
    return int(delta_x), int(delta_y)


def alignRobust2Bands(band_g, band_b):
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh(band_g, band_b,
                                                         surf_th,
                                                         surf_th_d,
                                                         kps_min,
                                                         kps_max)

    kps_gb_g = []
    kps_gb_b = []
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORB(band_g, band_b, orb_th)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFT(band_g, band_b, sift_th)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    return kps_gb_g, kps_gb_b, img


def alignRobust2BandsCloud(band_g, band_b):
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh_Cloud(band_g, band_b,
                                                               surf_th,
                                                               surf_th_d,
                                                               kps_min,
                                                               kps_max)
    # kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURFCloud(band_g, band_b, surf_th)

    kps_gb_g = []
    kps_gb_b = []
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORBCloud(band_g, band_b, orb_th)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFTCloud(band_g, band_b, sift_th)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    return kps_gb_g, kps_gb_b, img


def alignRobust2BandsGlobal(band_g, band_b):
    kps_gb_g = []
    kps_gb_b = []

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh(band_g, band_b,
                                                         surf_th_global,
                                                         surf_th_d_global,
                                                         kps_min_global,
                                                         kps_max_global)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORB(band_g, band_b, orb_th_global)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFT(band_g, band_b, sift_th_global)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    return kps_gb_g, kps_gb_b, img


def alignRobust2BandsGlobalCloud(band_g, band_b):
    kps_gb_g = []
    kps_gb_b = []

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh_Cloud(band_g, band_b,
                                                               surf_th_global,
                                                               surf_th_d_global,
                                                               kps_min_global,
                                                               kps_max_global)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORBCloud(band_g, band_b, orb_th_global)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFTCloud(band_g, band_b, sift_th_global)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    return kps_gb_g, kps_gb_b, img


def alignRobust2BandsTIF(band_g, band_b, counter, block):
    kps_gb_g = []
    kps_gb_b = []

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh(band_g, band_b, surf_th, surf_th_d, kps_min, kps_max)
    if isDebugMode:
        cv2.imwrite("output/surf_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                    img)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORB(band_g, band_b, orb_th)
        if isDebugMode:
            cv2.imwrite("output/orb_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                        img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFT(band_g, band_b, sift_th)
        if isDebugMode:
            cv2.imwrite(
                "output/sift_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # 输出最终匹配结果到txt文件
    if isDebugMode:
        fout = open("output/match_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".txt",
                    'w')
        for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
            fout.write(
                kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[1].__str__() + "\n")
        fout.close()
    return kps_gb_g, kps_gb_b


def alignRobust2BandsTIFCloud(band_g, band_b, counter, block):
    kps_gb_g = []
    kps_gb_b = []

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh_Cloud(band_g, band_b, surf_th, surf_th_d, kps_min, kps_max)
    if isDebugMode:
        cv2.imwrite("output/surf_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                    img)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORBCloud(band_g, band_b, orb_th)
        if isDebugMode:
            cv2.imwrite("output/orb_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                        img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFTCloud(band_g, band_b, sift_th)
        if isDebugMode:
            cv2.imwrite(
                "output/sift_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # 输出最终匹配结果到txt文件
    if isDebugMode:
        fout = open("output/match_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".txt",
                    'w')
        for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
            fout.write(
                kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[1].__str__() + "\n")
        fout.close()
    return kps_gb_g, kps_gb_b


def alignRobust2BandsTIFWithStretch(band_g, band_b, counter, block):
    kps_gb_g = []
    kps_gb_b = []

    # 分条带单独线性拉伸
    band_g = linearStretch(band_g, 1, 255, 0.02)
    band_b = linearStretch(band_b, 1, 255, 0.02)

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh(band_g, band_b, surf_th, surf_th_d, kps_min, kps_max)
    if isDebugMode:
        cv2.imwrite("output/surf_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                    img)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORB(band_g, band_b, orb_th)
        if isDebugMode:
            cv2.imwrite("output/orb_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                        img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFT(band_g, band_b, sift_th)
        if isDebugMode:
            cv2.imwrite(
                "output/sift_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # 输出最终匹配结果到txt文件
    if isDebugMode:
        fout = open("output/match_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".txt",
                    'w')
        for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
            fout.write(
                kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[1].__str__() + "\n")
        fout.close()
    return kps_gb_g, kps_gb_b


def alignRobust2BandsTIFWithStretchCloud(band_g, band_b, counter, block):
    kps_gb_g = []
    kps_gb_b = []

    # 分条带单独线性拉伸
    band_g = linearStretch(band_g, 1, 255, 0.02)
    band_b = linearStretch(band_b, 1, 255, 0.02)

    # surf匹配
    kp_gb1, kp_gb2, gb_matches1, img = FLANN_SURF_AutoTh_Cloud(band_g, band_b, surf_th, surf_th_d, kps_min, kps_max)
    if isDebugMode:
        cv2.imwrite("output/surf_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                    img)
    # 匹配输出的list不会为none，但有可能size为0，所以需要判断一下
    if kp_gb1.__len__() != 0:
        kps_gb_g.extend(kp_gb1)
        kps_gb_b.extend(kp_gb2)

    # orb匹配
    if kps_gb_g.__len__() < 3:
        print("match points less than 3,try to add ORB features.")
        kps1, kps2, matches, img = BF_ORB(band_g, band_b, orb_th)
        if isDebugMode:
            cv2.imwrite("output/orb_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                        img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # sift匹配
    if kps_gb_g.__len__() < 3:
        print("match points is still less than 3,try to add SIFT features.")
        kps1, kps2, matches, img = FLANN_SIFT(band_g, band_b, sift_th)
        if isDebugMode:
            cv2.imwrite(
                "output/sift_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".jpg",
                img)
        if kps1.__len__() != 0:
            kps_gb_g.extend(kps1)
            kps_gb_b.extend(kps2)

    # 输出最终匹配结果到txt文件
    if isDebugMode:
        fout = open("output/match_band_" + counter.__str__().zfill(2) + "_block_" + block.__str__().zfill(2) + ".txt",
                    'w')
        for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
            fout.write(
                kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[1].__str__() + "\n")
        fout.close()
    return kps_gb_g, kps_gb_b


def writeTif(bands, path):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")


def readTifImage(img_path):
    data = []
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return data
    else:
        print("Open image file success.")
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")
        for i in range(bands_num):
            # 获取影像的第i+1个波段
            band_i = dataset.GetRasterBand(i + 1)
            # 读取第i+1个波段数据
            band_data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
            data.append(band_data)
            print("band " + (i + 1).__str__() + " read success.")
        return data


def readTifImageWithWindow(img_path, start_x, start_y, x_range, y_range):
    data = []
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return data
    else:
        print("Open image file success.")
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")
        for i in range(bands_num):
            # 获取影像的第i+1个波段
            band_i = dataset.GetRasterBand(i + 1)
            # 读取第i+1个波段数据
            band_data = band_i.ReadAsArray(start_x, start_y, x_range, y_range)
            data.append(band_data)
            print("band " + (i + 1).__str__() + " read success.")
        return data


def writeTiff(path, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_bands = 1
        im_height, im_width = im_data.shape
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset


def loadImageWithWindow(img_path, start_x, start_y, x_range, y_range):
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    band_1 = dataset.GetRasterBand(1)
    data = band_1.ReadAsArray(start_x, start_y, x_range, y_range)
    return data


def cloudFilter(img, kps, des, ksize=5, iter=2):
    print(img.shape.__len__())
    if img.shape.__len__() != 2:
        cloud = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        cloud = img
    # 二值化
    if cloud_th == -1:
        ret1, th = cv2.threshold(cloud, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Auto threshold:" + ret1.__str__())
    else:
        ret1, th = cv2.threshold(cloud, cloud_th, 255, cv2.THRESH_BINARY)

    # 膨胀
    kernel = np.ones((ksize, ksize), np.uint8)
    dilate = cv2.dilate(th, kernel, iterations=iter)

    # 生成掩膜
    mask = cv2.inRange(dilate, 0, 0)
    dst = cv2.bitwise_and(cloud, cloud, mask=mask)
    # dst = linearStretch(dst, 1, 255, 0.02)
    # dst_rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    # 根据掩膜过滤特征点
    good_kps = []
    good_des = []
    removed_kps = []
    removed_des = []
    for i in range(kps.__len__()):
        if mask[int(kps[i].pt[1]), int(kps[i].pt[0])] == 255:
            good_kps.append(kps[i])
            good_des.append(des[i])
        else:
            removed_kps.append(kps[i])
            removed_des.append(des[i])
    good_des = np.array(good_des)

    # 绘制过滤后的特征点
    cloud = cv2.cvtColor(cloud, cv2.COLOR_GRAY2BGR)
    img_new = drawKeyPoints(cloud, good_kps, color=(0, 255, 0))
    img_new = drawKeyPoints(img_new, removed_kps, color=(0, 0, 255))
    print("input points:" + kps.__len__().__str__())
    print("filtered points:" + good_kps.__len__().__str__())
    return good_kps, good_des, removed_kps, removed_des, mask, img_new


def getStripeRange(imgName1, imgName2,
                   band_b_ori, band_g_ori, band_b_10, band_g_10,
                   img_h_b, img_h_g, img_w_g,
                   gb_dx, gb_dy,
                   isStripeStretch,
                   imgIndex,
                   kp1, kp2):
    stripe_items = []
    min_height = min(img_h_b, img_h_g)
    stripes = min_height / stripe_height
    default_affine, mask = cv2.estimateAffine2D(np.array(kp1), np.array(kp2))
    print(default_affine)
    for i in range(stripes):
        gb_resample_start_y = i * stripe_height - stripe_extension + gb_dy
        gb_resample_end_y = i * stripe_height + stripe_height + stripe_extension + gb_dy

        if gb_resample_start_y < 0:
            gb_resample_start_y = 0
        if gb_resample_end_y > min_height:
            gb_resample_end_y = min_height
            # 有时会出现这种情况，如1000,2000，各加1500变成2500,3500，但图像为2200，所以3500超过2200，变为2200
            # 这样就会出现2500-2200这样的情况，从而导致打开图像失败
            if gb_resample_start_y > gb_resample_end_y:
                gb_resample_start_y = min_height - stripe_height - 2 * stripe_extension

        b_start_y = gb_resample_start_y
        b_end_y = gb_resample_end_y
        g_start_y = i * stripe_height
        g_end_y = i * stripe_height + stripe_height
        band_b_stripe_ori = band_b_ori[b_start_y:b_end_y, :]
        band_g_stripe_ori = band_g_ori[g_start_y:g_end_y, :]
        band_b_stripe_res = band_b_10[b_start_y:b_end_y, :]
        band_g_stripe_res = band_g_10[g_start_y:g_end_y, :]

        # 一条记录中保存相关信息
        stripe_items.append((imgName1, imgName2,
                             b_start_y, b_end_y, g_start_y, g_end_y,
                             band_b_stripe_ori, band_g_stripe_ori,
                             band_b_stripe_res, band_g_stripe_res,
                             i + 1,
                             img_h_b, img_h_g, img_w_g,
                             isStripeStretch, imgIndex,
                             default_affine))

    return stripe_items


def resampleStripe(stripe_input):
    # 并行函数需要去除所有与顺序有关的内容，如索引、如相邻仿射矩阵约束等，否则这样的设计就没法并行
    t1 = time.time()

    img_name1 = stripe_input[0]
    img_name2 = stripe_input[1]
    b_start_y = stripe_input[2]
    b_end_y = stripe_input[3]
    g_start_y = stripe_input[4]
    g_end_y = stripe_input[5]
    band_b = stripe_input[6]
    band_g = stripe_input[7]
    band_b_resample = stripe_input[8]
    band_g_resample = stripe_input[9]
    i = stripe_input[10]
    img_h_b = stripe_input[11]
    img_h_g = stripe_input[12]
    img_w_g = stripe_input[13]
    isStripeStretch = stripe_input[14]
    it = stripe_input[15]
    default_affine = stripe_input[16]

    print("band base:" + g_start_y.__str__() + " " + g_end_y.__str__())
    print("band resample:" + b_start_y.__str__() + " " + b_end_y.__str__())

    # 用于判断基准影像是否为空，对应于当基准影像的黑色部分超过条带高度时，会引发重采异常，导致多复制影像
    res = np.count_nonzero(band_g_resample)
    if (res * 1.0) / (band_g_resample.shape[0] * band_g_resample.shape[1]) < 0.2:
        print("base image is empty.")
        resampled_band_b = np.zeros([stripe_height, img_w_g], np.uint16)
        affine1 = np.array([[1, 0, 0],
                            [0, 1, 0]])

        t2 = time.time()
        dt = t2 - t1
        print("cost time:" + dt.__str__())
        return resampled_band_b, dt, affine1

    # 分条带二次拉伸、配准
    if isStripeStretch:
        if isCloudMode:
            kps_gb_g, kps_gb_b = alignRobust2BandsTIFWithStretchCloud(band_g, band_b, it + 1, i + 1)
        else:
            kps_gb_g, kps_gb_b = alignRobust2BandsTIFWithStretch(band_g, band_b, it + 1, i + 1)
    else:
        if isCloudMode:
            kps_gb_g, kps_gb_b = alignRobust2BandsTIFCloud(band_g, band_b, it + 1, i + 1)
        else:
            kps_gb_g, kps_gb_b = alignRobust2BandsTIF(band_g, band_b, it + 1, i + 1)

    # 当匹配点对小于3，无法构建仿射矩阵
    if kps_gb_g.__len__() < 3:
        # 判断上一个条带是否成功生成仿射矩阵，否的话直接复制影像结束本次循环
        print("KeyPoint size is less than 3,try to use default affine mat.")
        if isDebugMode:
            cv2.imwrite("output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                        band_b)
        # 重采
        resampled_band_b = cv2.warpAffine(band_b_resample, default_affine,
                                          (band_g_resample.shape[1], band_g_resample.shape[0]))
        t2 = time.time()
        dt = t2 - t1
        print("cost time:" + dt.__str__())
        return resampled_band_b, dt, default_affine
    # 匹配点对大于3，继续判断能否生成仿射矩阵以及生成的是否正确
    else:
        affine1, mask = cv2.estimateAffine2D(np.array(kps_gb_b), np.array(kps_gb_g))
        # 如果生成失败，判断上一个条带是否成功生成仿射矩阵，如果是的话，使用上一个矩阵，否的话直接复制影像结束本次循环
        if affine1 is None:
            print("Estimated affine matrix is none, try to use default affine mat.")
            if isDebugMode:
                cv2.imwrite("output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                            band_b)
            # 重采
            resampled_band_b = cv2.warpAffine(band_b_resample, default_affine,
                                              (band_g_resample.shape[1], band_g_resample.shape[0]))
            t2 = time.time()
            dt = t2 - t1
            print("cost time:" + dt.__str__())
            return resampled_band_b, dt, default_affine
        # affine matrix不为空，再判断是否正确
        else:
            num1 = affine1[0][0]
            num2 = affine1[1][1]
            print(num1, num2)
            if affine_min < num1 < affine_max and affine_min < num2 < affine_max:
                # 重采
                resampled_band_b = cv2.warpAffine(band_b_resample, affine1,
                                                  (band_g_resample.shape[1], band_g_resample.shape[0]))
                t2 = time.time()
                dt = t2 - t1
                print("cost time:" + dt.__str__())
                return resampled_band_b, dt, affine1
            else:
                print("Estimated affine matrix is wrong, try to use default affine mat.")
                if isDebugMode:
                    cv2.imwrite(
                        "output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                        band_b)
                # 重采
                resampled_band_b = cv2.warpAffine(band_b_resample, default_affine,
                                                  (band_g_resample.shape[1], band_g_resample.shape[0]))
                t2 = time.time()
                dt = t2 - t1
                print("cost time:" + dt.__str__())
                return resampled_band_b, dt, default_affine
