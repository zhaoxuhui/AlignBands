# coding=utf-8
import sys
import cv2
import functions as fun
import os

if __name__ == '__main__':
    if sys.argv.__len__() >= 2:
        if sys.argv[1] == 'help' or sys.argv[1] == 'HELP':
            print("Function description:")
            print("Join several bands into one file.")
            print("\nUsage instruction:")
            print("example.exe [img_dir] [img_type] [out_path]")
            print("[img_dir]:The input dir that contains band data.")
            print("[img_type]:The file type of band data,tif or png etc.")
            print("[out_path]:The filename of joined image.")
            print("Please note that these band data should have same height and width.")
            print("\nUsage example:")
            print("Tool_JoinBands.exe C:\\tif tif C:\\tifout\\joined.tif")
            os.system('pause')
        else:
            img_dir = sys.argv[1]
            img_type = sys.argv[2]
            out_path = sys.argv[3]
            paths, names, files = fun.findAllFiles(img_dir, img_type)
            bands_data = []

            # 对于tif文件，统一用gdal打开并输出为tif文件
            if img_type.endswith('tif') or img_type.endswith('TIF') or img_type.endswith('TIFF') or img_type.endswith(
                    'tiff'):
                for i in range(files.__len__()):
                    band_data = fun.readTifImage(files[i])
                    bands_data.extend(band_data)
                    print("joined " + (i + 1).__str__() + " bands.")
                print(bands_data.__len__().__str__() + " bands in total.")
                fun.writeTif(bands_data, out_path)
            # 对于所有其它类型的文件，如jpg、png等，统一用OpenCV处理
            else:
                for i in range(files.__len__()):
                    band_data = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
                    bands_data.append(band_data)
                print("Open image success.")
                data = cv2.merge((bands_data[0], bands_data[1], bands_data[2]))
                cv2.imwrite(out_path, data)
                print("Save image success.")
    else:
        print("Unknown mode, input 'yourExeName.exe help' to get help information.")
