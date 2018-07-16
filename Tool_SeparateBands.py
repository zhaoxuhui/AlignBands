# coding=utf-8
import sys
import os
import cv2
import functions as fun

if __name__ == '__main__':
    if sys.argv.__len__() >= 2:
        if sys.argv[1] == 'help' or sys.argv[1] == 'HELP':
            print("Function description:")
            print("Separate and save different band data in one image file.")
            print("\nUsage instruction:")
            print("example.exe [img_path] [out_dir]")
            print("[img_path]:The filename of input image.")
            print("[output_dir]:The output dir for different band images.")
            print("\nUsage example:")
            print("Tool_SeparateBands.exe C:\\tif\\input.tif C:\\tifout")
            os.system('pause')

        else:
            img_path = sys.argv[1]
            output_dir = sys.argv[2]

            # 对于tif文件，统一用gdal打开并输出为tif文件
            if img_path.endswith('tif') or img_path.endswith('TIF') or img_path.endswith('TIFF') or img_path.endswith(
                    'tiff'):
                bands_data = fun.readTifImage(img_path)
                for i in range(bands_data.__len__()):
                    fun.writeTif([bands_data[i]], output_dir + os.path.sep + "band_" + i.__str__().zfill(2) + ".tif")
                    print("saved " + (i + 1).__str__() + "/" + bands_data.__len__().__str__())
            # 对于所有其它类型的文件，如jpg、png等，统一用OpenCV处理
            else:
                img = cv2.imread(img_path)
                print("Open image success.")
                band_b, band_g, band_r = cv2.split(img)
                cv2.imwrite(output_dir + os.path.sep + "band_b.png", band_b)
                cv2.imwrite(output_dir + os.path.sep + "band_g.png", band_g)
                cv2.imwrite(output_dir + os.path.sep + "band_r.png", band_r)
                print("Save image success.")

    else:
        print("Unknown mode, input 'yourExeName.exe help' to get help information.")
