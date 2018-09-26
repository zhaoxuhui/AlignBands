# coding=utf-8
import shutil
import cv2
import sys
import functions as fun
import time
import numpy as np
import os
from multiprocessing import Pool

if __name__ == '__main__':
    if sys.argv.__len__() == 2 and sys.argv[1] == 'help':
        print("Function description:")
        print("Read bands(tif file) and align with stripe mode."
              "This mode support removing the movement of clouds so that align results will be more accurate.")
        print("\nUsage instruction:")
        print("example.exe [img_dir] [img_out_dir]")
        print("[img_dir]:The input dir that contains band data(tif files).")
        print("[img_out_dir]:The output dir for resampled images.")
        print("\nUsage example:")
        print("AlignTIF_Stripe.exe C:\\tif C:\\tifout")
        os.system('pause')
    elif sys.argv.__len__() == 3:
        # 多波段分条带对准模式（tif -> tif）
        print('---Stripe mode for tif files---')
        # 读取参数
        fun.readConfigFile('config.yml')

        if fun.isCloudMode:
            print("=>Cloud movement detector on")
        else:
            print("=>Cloud movement detector off")
        exe_paths = []
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
        cost_time = []

        flag = raw_input("Continue?y/n")
        if flag == 'y':
            for it in range(exe_paths.__len__()):
                t1 = time.time()
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

                # 2%灰度拉伸
                band_b_ori, old_min_b, old_max_b = fun.linearStretchWithData(
                    np.uint8(np.where(band_b_10 == 0, 128, band_b_10) / 4), 1, 255, 0.02)
                band_g_ori, old_min_g, old_max_g = fun.linearStretchWithData(
                    np.uint8(np.where(band_g_10 == 0, 128, band_g_10) / 4), 1, 255, 0.02)
                # 判断是全局拉伸就可以了还是需要分条带拉伸，阈值100
                if (old_max_b - old_min_b) > fun.stretch_th or (old_max_g - old_min_g) > fun.stretch_th:
                    isStripeStretch = True
                    print("Grayscale stretch for every stripe.")
                else:
                    isStripeStretch = False

                gb_dx, gb_dy, kp1, kp2 = fun.getBandsOffsetWithNoStretchWithData(band_b_ori, band_g_ori, (it + 1))
                yoffsets.append(gb_dy)

                img_parts = []
                affine_matrices_gb = []

                img_h_b = band_b_ori.shape[0]
                img_h_g = band_g_ori.shape[0]
                img_w_g = band_g_ori.shape[1]

                stripe_input = fun.getStripeRange(img_name1, img_name2,
                                                  band_b_ori, band_g_ori,
                                                  band_b_10, band_g_10,
                                                  img_h_b, img_h_g, img_w_g,
                                                  gb_dx, gb_dy,
                                                  isStripeStretch,
                                                  it,
                                                  kp1, kp2)

                stripe_output = []
                # 新建指定数量的进程池用于对多进程进行管理
                pool = Pool(processes=fun.processNum)
                # 注意map函数中传入的参数应该是可迭代对象，如list
                stripe_output = pool.map(fun.resampleStripe, stripe_input)
                pool.close()
                pool.join()


                # 拆解并行处理打包输出的数据
                for item in stripe_output:
                    img_parts.append(item[0])
                    affine_matrices_gb.append(item[2])

                # 剩余部分处理，保证重采后的图与原图大小相同
                print("\nDeal with residual part...")
                residual_part_g = np.zeros([img_h_g - fun.stripe_height * stripe_output.__len__(), img_w_g], np.uint16)
                img_parts.append(residual_part_g)

                t2 = time.time()
                cost_time.append(t2 - t1)
                print("Total cost time:" + (t2 - t1).__str__() + " s")
                img_out = img_parts[0]
                print("\nMosaic images...")
                for i in range(1, img_parts.__len__()):
                    img_out = np.vstack((img_out, img_parts[i]))
                print("Save image...")
                # 分别保存tif和jpg以应对不同用途
                img_out_8 = np.uint8(img_out / 4)
                cv2.imwrite(out_path.replace('.tif', '.jpg'), img_out_8)
                fun.writeTiff(out_path, img_out)
                print("Success.")

            file_out = open(output_path + "\\y_offset.txt", 'w')
            for item in yoffsets:
                file_out.write(item.__str__() + "\n")
            file_out.close()
            if fun.isIteration:
                # copy base image
                shutil.copy(files[0], output_path + "\\" + names[0])
            print("Band align finished.")
            print("All bands total cost time:" + sum(cost_time).__str__() + " s")
            os.system('pause')
        else:
            os.system('pause')
    else:
        print("Input 'yourExeName.exe help' to get help information.")
        os.system('pause')
