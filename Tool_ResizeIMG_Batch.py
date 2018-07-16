# coding=utf-8
import sys
import cv2
import functions as fun
import os

if __name__ == '__main__':
    if sys.argv.__len__() >= 2:
        if sys.argv[1] == 'help' or sys.argv[1] == 'HELP':
            print("Function description:")
            print("Select and cut the ROI(region of interest) in big image files(Batch mode).")
            print("\nUsage instruction:")
            print("example.exe [img_dir] [img_type] [output_dir] [start_x] [start_y] [x_range] [y_range]")
            print("[img_dir]:The input dir that contains band data.")
            print("[img_type]:The file type of band data,tif or png etc.")
            print("[output_dir]:The output dir for ROI images.")
            print("[start_x]:The x coordinate of ROI's left-top point in big image.")
            print("[start_y]:The y coordinate of ROI's left-top point in big image.")
            print("[x_range]:The range of ROI in x direction(width).")
            print("[y_range]:The range of ROI in y direction(height).")
            print("\nUsage example:")
            print("Tool_ResizeIMG_Batch.exe C:\\tif tif C:\\tifout 100 200 3000 4000")
            os.system('pause')
        else:
            img_dir = sys.argv[1]
            img_type = sys.argv[2]
            out_dir = sys.argv[3]
            start_x = int(sys.argv[4])
            start_y = int(sys.argv[5])
            x_range = int(sys.argv[6])
            y_range = int(sys.argv[7])

            paths, names, files = fun.findAllFiles(img_dir, img_type)
            # 对于tif文件，统一用gdal打开并输出为tif文件
            if img_type.endswith('tif') or img_type.endswith('TIF') or img_type.endswith('TIFF') or img_type.endswith(
                    'tiff'):
                for i in range(files.__len__()):
                    bands_data = fun.readTifImageWithWindow(files[i], start_x, start_y, x_range, y_range)
                    fun.writeTif(bands_data, out_dir + os.path.sep + names[i][:names[i].rfind('.')] + "_cut.tif")
                    print("cutting " + (i + 1).__str__() + "/" + files.__len__().__str__())
                print('cut finished.')
            # 对于所有其它类型的文件，如jpg、png等，统一用OpenCV处理
            else:
                for i in range(files.__len__()):
                    bands_data = cv2.imread(files[i])
                    bands_data_roi = bands_data[start_y:start_y + y_range, start_x:start_x + x_range, :]
                    cv2.imwrite(out_dir + os.path.sep + "band_" + (i + 1).__str__().zfill(2) + ".jpg", bands_data_roi)
                    print("cutting " + (i + 1).__str__() + "/" + files.__len__())
                print('cut finished.')
    else:
        print("Unknown mode, input 'yourExeName.exe help' to get help information.")
