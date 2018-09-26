# coding=utf-8
import shutil
import cv2
import sys
import functions as fun
import time
import numpy as np
import os

if __name__ == '__main__':
    if sys.argv.__len__() == 2 and sys.argv[1] == 'help':
        print("Function description:")
        print("Read bands(tif file) and align with global mode."
              "This mode support removing the movement of clouds so that align results will be more accurate.")
        print("\nUsage instruction:")
        print("example.exe [img_dir] [img_out_dir]")
        print("[img_dir]:The input dir that contains band data(tif files).")
        print("[img_out_dir]:The output dir for resampled images.")
        print("Please note that the height or width of input image should not exceed 32000 pixels.")
        print("\nUsage example:")
        print("AlignTIF_IMG.exe C:\\tif C:\\tifout")
        os.system('pause')
    elif sys.argv.__len__() == 3:
        # 全图对准模式（tif -> tif）
        print('---Global mode for tif files---')
        # 读取参数
        fun.readConfigFile('config.yml')

        exe_paths = []
        if fun.isCloudMode:
            print("=>Cloud movement detector on")
        else:
            print("=>Cloud movement detector off")
        if fun.isIteration:
            print('---Base image iteration mode---')
            # 构造执行命令
            input_path = sys.argv[1]
            output_path = sys.argv[2]
            if fun.isReverse:
                paths, names, files = fun.findAllFilesReverse(input_path, '.tif')
            else:
                paths, names, files = fun.findAllFiles(input_path, '.tif')
            for item in files:
                print(item)
            res = fun.generateOutputFilename(output_path, names, filetype='tif')
            print("Images going to output:")
            for item in res:
                print(item)

            for i in range(res.__len__()):
                if i % 2 == 0:
                    if i == 0:
                        exe_paths.append([files[i], files[i + 1], res[i]])
                    else:
                        exe_paths.append([res[i - 1], files[i + 1], res[i]])
                else:
                    exe_paths.append([res[i - 1], files[i + 1], res[i]])
        else:
            print('---Fixed base image mode---')
            base_img_path = sys.argv[1]
            print("Base image:" + base_img_path)
            input_dir = base_img_path[:base_img_path.rfind(os.path.sep)]
            output_path = sys.argv[2]
            paths, names, files = fun.findAllFiles(input_dir, '.tif')

            for name, file in zip(names, files):
                if base_img_path == file:
                    exe_paths.append([base_img_path, file,
                                      output_path + os.path.sep + name[:name.rfind('.')] + '_base.tif'])
                else:
                    exe_paths.append([base_img_path, file,
                                      output_path + os.path.sep + name[:name.rfind('.')] + '_out.tif'])

        print("Combinations of images:")
        for item in exe_paths:
            print(item)
        yoffsets = []

        flag = raw_input("Continue?y/n")
        if flag == 'y':
            for it in range(exe_paths.__len__()):
                path_g = exe_paths[it][0]
                path_b = exe_paths[it][1]
                out_path = exe_paths[it][2]

                if fun.isIteration is False:
                    if path_g == path_b:
                        # copy base image
                        print("\nThis is base image,copying base image to des dir...")
                        shutil.copy(path_g, out_path)
                        continue

                img_name1 = path_g[path_g.rfind("\\") + 1:]
                img_name2 = path_b[path_b.rfind("\\") + 1:]
                print("\nBand " + (it + 1).__str__() + "/" + exe_paths.__len__().__str__())
                print("Data Info:")
                print("Base image:" + path_g)
                print("Resample image:" + path_b)
                print("Out image:" + out_path)

                band_b_10 = fun.loadIMGintoMem(path_b)
                band_g_10 = fun.loadIMGintoMem(path_g)

                # 去除纯黑噪音块
                band_b, old_min_b, old_max_b = fun.linearStretchWithData(
                    np.uint8(np.where(band_b_10 == 0, 128, band_b_10) / 4), 1, 255, 0.02)
                band_g, old_min_g, old_max_g = fun.linearStretchWithData(
                    np.uint8(np.where(band_g_10 == 0, 128, band_g_10) / 4), 1, 255, 0.02)

                # # 去除白云
                # band_b, old_min_b, old_max_b = fun.linearStretchWithData(
                #     np.uint8(np.where(band_b > 140, 128, band_b) / 4), 1, 255, 0.02)
                # band_g, old_min_g, old_max_g = fun.linearStretchWithData(
                #     np.uint8(np.where(band_g > 140, 128, band_g) / 4), 1, 255, 0.02)

                cost_time = []

                t1 = time.time()

                band_b_resample = band_b_10
                band_g_resample = band_g_10

                if fun.isCloudMode:
                    kps_gb_g, kps_gb_b, img_match = fun.alignRobust2BandsGlobalCloud(band_g, band_b)
                else:
                    kps_gb_g, kps_gb_b, img_match = fun.alignRobust2BandsGlobal(band_g, band_b)
                if fun.isDebugMode:
                    cv2.imwrite("output/match_" + img_name1 + "_" + img_name2 + ".jpg", img_match)
                    fout = open("output/match_" + img_name1 + "_" + img_name2 + ".txt", 'w')
                    for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
                        fout.write(
                            kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[
                                1].__str__() + "\n")
                    fout.close()

                # 当匹配点对小于3，无法构建仿射矩阵
                if kps_gb_g.__len__() < 3:
                    fun.writeTiff(out_path, band_b_10)
                    continue
                # 匹配点对大于3，继续判断能否生成仿射矩阵以及生成的是否正确
                else:
                    affine1, mask = cv2.estimateAffine2D(np.array(kps_gb_b), np.array(kps_gb_g))
                    # 如果生成失败，判断上一个条带是否成功生成仿射矩阵，如果是的话，使用上一个矩阵，否的话直接复制影像结束本次循环
                    if affine1 is None:
                        fun.writeTiff(out_path, band_b_10)
                        continue
                    # affine matrix不为空，再判断是否正确
                    else:
                        num1 = affine1[0][0]
                        num2 = affine1[1][1]
                        print(num1, num2)
                        if fun.affine_min < num1 < fun.affine_max and fun.affine_min < num2 < fun.affine_max:
                            print(affine1)
                            resampled_band_b = cv2.warpAffine(band_b_resample, affine1,
                                                              (band_g_resample.shape[1],
                                                               band_g_resample.shape[0]))
                        else:
                            fun.writeTiff(out_path, band_b_10)
                            continue

                t2 = time.time()
                dt = t2 - t1
                cost_time.append(dt)
                print("cost time:" + dt.__str__())

                print("Total cost time:" + sum(cost_time).__str__() + " s")
                print("Save image...")
                # 分别保存tif和jpg以应对不同用途
                img_out_8 = np.uint8(resampled_band_b / 4)
                cv2.imwrite(out_path.replace('.tif', '.jpg'), img_out_8)
                fun.writeTiff(out_path, resampled_band_b)
                print("Success.")

            file_out = open(output_path + "\\y_offset.txt", 'w')
            for item in yoffsets:
                file_out.write(item.__str__() + "\n")
            file_out.close()
            if fun.isIteration:
                # copy base image
                shutil.copy(files[0], output_path + "\\" + names[0])
            print("Band align finished.")
            os.system('pause')
        else:
            os.system('pause')
    else:
        print("Input 'yourExeName.exe help' to get help information.")
        os.system('pause')
