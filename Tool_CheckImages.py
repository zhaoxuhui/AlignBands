# coding=utf-8
import os
import cv2
import sys
import argparse


def getPreviewOneDay(root_dir, out_dir, band_r, band_g, band_b, dtype):
    files = os.listdir(root_dir)
    files.sort()
    filtered_files = []
    for file in files:
        if file[0] == "H":
            filtered_files.append(root_dir + os.sep + file)

    for i in range(filtered_files.__len__()):
        current = i + 1
        total = filtered_files.__len__()
        print filtered_files[i]
        getPreview(filtered_files[i], out_dir, band_r, band_g, band_b, current, total, dtype)


def getPreview(root_dir, out_dir, band_r, band_g, band_b, current, total, dtype):
    img_uid = root_dir.split('\\')[-1]
    print img_uid

    img_nums = []
    files = os.listdir(root_dir)
    files.sort()
    for item in files:
        img_num_dir = root_dir + os.sep + item
        img_nums.append((item, img_num_dir))

    ccd_nums = []
    for i in range(img_nums.__len__()):
        for j in range(3):
            ccd_num = img_nums[i][1] + os.sep + img_nums[i][0] + "_CCD" + (j + 1).__str__()
            if os.path.exists(ccd_num):
                ccd_nums.append((img_nums[i][0], ccd_num))

    rgb_files = []
    for i in range(ccd_nums.__len__()):
        if ccd_nums[i][1].endswith("CCD1"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD1." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD1." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD1." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))
        elif ccd_nums[i][1].endswith("CCD2"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD2." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD2." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD2." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))
        elif ccd_nums[i][1].endswith("CCD3"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD3." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD3." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD3." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))

    for i in range(rgb_files.__len__()):
        band_r = cv2.imread(rgb_files[i][0], cv2.IMREAD_GRAYSCALE)
        band_g = cv2.imread(rgb_files[i][1], cv2.IMREAD_GRAYSCALE)
        band_b = cv2.imread(rgb_files[i][2], cv2.IMREAD_GRAYSCALE)
        bands = cv2.merge((band_b, band_g, band_r))
        cv2.imwrite(
            out_dir + os.sep + img_uid + "_" + rgb_files[i][3] + "-" + rgb_files[i][4][10:] + "-" + rgb_files[i][5][
                                                                                                    10:] + "." + dtype,
            bands)
        print "\n" + (
            i + 1).__str__() + "/" + rgb_files.__len__().__str__() + "," + current.__str__() + "/" + total.__str__()
        print img_uid + "_" + rgb_files[i][3]
        print img_uid + "_" + rgb_files[i][4]
        print img_uid + "_" + rgb_files[i][5]


def getPreviewParticular(root_dir, out_dir, band_r, band_g, band_b, dtype):
    img_uid = root_dir.split('\\')[-1]
    print img_uid

    img_nums = []
    files = os.listdir(root_dir)
    files.sort()
    for item in files:
        img_num_dir = root_dir + os.sep + item
        img_nums.append((item, img_num_dir))

    ccd_nums = []
    for i in range(img_nums.__len__()):
        for j in range(3):
            ccd_num = img_nums[i][1] + os.sep + img_nums[i][0] + "_CCD" + (j + 1).__str__()
            if os.path.exists(ccd_num):
                ccd_nums.append((img_nums[i][0], ccd_num))

    rgb_files = []
    for i in range(ccd_nums.__len__()):
        if ccd_nums[i][1].endswith("CCD1"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD1." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD1." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD1." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))
        elif ccd_nums[i][1].endswith("CCD2"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD2." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD2." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD2." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))
        elif ccd_nums[i][1].endswith("CCD3"):
            r_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_r + "_CCD3." + dtype
            g_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_g + "_CCD3." + dtype
            b_file = ccd_nums[i][0].replace("_L1", "_L1B") + "_B" + band_b + "_CCD3." + dtype
            r_file2 = ccd_nums[i][1] + os.sep + r_file
            g_file2 = ccd_nums[i][1] + os.sep + g_file
            b_file2 = ccd_nums[i][1] + os.sep + b_file
            if os.path.exists(r_file2) and os.path.exists(g_file2) and os.path.exists(b_file2):
                rgb_files.append((r_file2,
                                  g_file2,
                                  b_file2,
                                  r_file[20:24] + "_" + r_file[37:41] + "_" + r_file[33:36],
                                  g_file[20:24] + "_" + g_file[37:41] + "_" + g_file[33:36],
                                  b_file[20:24] + "_" + b_file[37:41] + "_" + b_file[33:36],))

    for i in range(rgb_files.__len__()):
        band_r = cv2.imread(rgb_files[i][0], cv2.IMREAD_GRAYSCALE)
        band_g = cv2.imread(rgb_files[i][1], cv2.IMREAD_GRAYSCALE)
        band_b = cv2.imread(rgb_files[i][2], cv2.IMREAD_GRAYSCALE)
        bands = cv2.merge((band_b, band_g, band_r))
        cv2.imwrite(
            out_dir + os.sep + img_uid + "_" + rgb_files[i][3] + "-" + rgb_files[i][4][10:] + "-" + rgb_files[i][5][
                                                                                                    10:] + "." + dtype,
            bands)
        print "\n" + (
            i + 1).__str__() + "/" + rgb_files.__len__().__str__()
        print img_uid + "_" + rgb_files[i][3]
        print img_uid + "_" + rgb_files[i][4]
        print img_uid + "_" + rgb_files[i][5]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for program.')
    parser.add_argument('-input', help='Input directory.')
    parser.add_argument('-output', default='.', help='Output directory.')
    parser.add_argument('-r', default='14', help='Red band number.')
    parser.add_argument('-g', default='07', help='Green band number.')
    parser.add_argument('-b', default='02', help='Blue band number.')
    parser.add_argument('-mode', default='d',
                        help='Different mode.s-one particular image directory;d-one day directory.')
    parser.add_argument('-type', default='jpg', help='Image type.jpg recommend.')

    args = parser.parse_args()
    if args.input is not None:
        root_dir = args.input
        out_dir = args.output
        band_r = args.r
        band_g = args.g
        band_b = args.b
        mode = args.mode
        dtype = args.type

        if mode == 'd':
            getPreviewOneDay(root_dir, out_dir, band_r, band_g, band_b, dtype)
        elif mode == 's':
            getPreviewParticular(root_dir, out_dir, band_r, band_g, band_b, dtype)
        else:
            print "Check input info."
            os.system('pause')
    else:
        print "No input images."
        os.system('pause')
