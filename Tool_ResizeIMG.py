# coding=utf-8
import sys
import cv2
import functions as fun
import os

if __name__ == '__main__':
    if sys.argv.__len__() >= 2:
        if sys.argv[1] == 'help' or sys.argv[1] == 'HELP':
            print("Function description:")
            print("Select and cut the ROI(region of interest) in a big image file.")
            print("\nUsage instruction:")
            print("example.exe [img_path] [out_path] [start_x] [start_y] [x_range] [y_range]")
            print("[img_path]:The filename of input image.")
            print("[out_path]:The filename of output image.")
            print("[start_x]:The x coordinate of ROI's left-top point in big image.")
            print("[start_y]:The y coordinate of ROI's left-top point in big image.")
            print("[x_range]:The range of ROI in x direction(width).")
            print("[y_range]:The range of ROI in y direction(height).")
            print("\nUsage example:")
            print("Tool_ResizeIMG.exe C:\\tif\\input.tif C:\\tifout\\roi.tif 100 200 3000 4000")
            os.system('pause')
        else:
            img_path = sys.argv[1]
            out_path = sys.argv[2]
            start_x = int(sys.argv[3])
            start_y = int(sys.argv[4])
            x_range = int(sys.argv[5])
            y_range = int(sys.argv[6])
            # 对于tif文件，统一用gdal打开并输出为tif文件
            if img_path.endswith('tif') or img_path.endswith('TIF') or img_path.endswith('TIFF') or img_path.endswith(
                    'tiff'):
                bands_data = fun.readTifImageWithWindow(img_path, start_x, start_y, x_range, y_range)
                fun.writeTif(bands_data, out_path)
            # 对于所有其它类型的文件，如jpg、png等，统一用OpenCV处理
            else:
                bands_data = cv2.imread(img_path)
                print("Open image success.")
                bands_data_roi = bands_data[start_y:start_y + y_range, start_x:start_x + x_range, :]
                cv2.imwrite(out_path, bands_data_roi)
                print("Save image success.")
    else:
        print("Unknown mode, input 'yourExeName.exe help' to get help information.")
