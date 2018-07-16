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
        print("Read bands(png file) and align with stripe mode."
              "This mode support removing the movement of clouds so that align results will be more accurate.")
        print("\nUsage instruction:")
        print("example.exe [img_dir] [img_out_dir]")
        print("[img_dir]:The input dir that contains band data(png files).")
        print("[img_out_dir]:The output dir for resampled images.")
        print("\nUsage example:")
        print("AlignPNG_Stripe.exe C:\\png C:\\pngout")
        os.system('pause')
    elif sys.argv.__len__() == 3:
        # 多波段分条带对准模式（png -> jpg）
        print('---Stripe mode for png files---')
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
                paths, names, files = fun.findAllFilesReverse(input_path, '.png')
            else:
                paths, names, files = fun.findAllFiles(input_path, '.png')
            for item in files:
                print(item)
            res = fun.generateOutputFilename(output_path, names, filetype='png')
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
            paths, names, files = fun.findAllFiles(input_dir, '.png')

            for name, file in zip(names, files):
                if base_img_path == file:
                    exe_paths.append([base_img_path, file,
                                      output_path + os.path.sep + name[:name.rfind('.')] + '_base.png'])
                else:
                    exe_paths.append([base_img_path, file,
                                      output_path + os.path.sep + name[:name.rfind('.')] + '_out.png'])

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

                band_b_ori = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
                band_g_ori = cv2.imread(path_g, cv2.IMREAD_GRAYSCALE)
                gb_dx, gb_dy = fun.getBandsOffsetWithNoStretch(band_b_ori, band_g_ori, (it + 1))
                yoffsets.append(gb_dy)

                img_parts = []
                cost_time = []
                affine_matrices_gb = []

                img_h_b = band_b_ori.shape[0]
                img_h_g = band_g_ori.shape[0]
                img_w_g = band_g_ori.shape[1]
                min_height = min(img_h_b, img_h_g)
                blocks = min_height / fun.stripe_height

                for i in range(blocks):
                    t1 = time.time()
                    print("\nBand " + (it + 1).__str__() + "/" + exe_paths.__len__().__str__() +
                          " stripe " + (i + 1).__str__() + "/" + blocks.__str__() +
                          "\n" + img_name1 + " " + img_name2)
                    gb_resample_start_y = i * fun.stripe_height - fun.stripe_extension + gb_dy
                    gb_resample_end_y = i * fun.stripe_height + fun.stripe_height + fun.stripe_extension + gb_dy

                    if gb_resample_start_y < 0:
                        gb_resample_start_y = 0
                    if gb_resample_end_y > min_height:
                        gb_resample_end_y = min_height
                        # 有时会出现这种情况，如1000,2000，各加1500变成2500,3500，但图像为2200，所以3500超过2200，变为2200
                        # 这样就会出现2500-2200这样的情况，从而导致打开图像失败
                        if gb_resample_start_y > gb_resample_end_y:
                            gb_resample_start_y = min_height - fun.stripe_height - 2 * fun.stripe_extension

                    band_b = band_b_ori[gb_resample_start_y:gb_resample_end_y, :]
                    band_g = band_g_ori[i * fun.stripe_height:i * fun.stripe_height + fun.stripe_height, :]
                    band_b_resample = band_b_ori[gb_resample_start_y:gb_resample_end_y, :]
                    band_g_resample = band_g_ori[i * fun.stripe_height:i * fun.stripe_height + fun.stripe_height, :]

                    print("band base:" + (i * fun.stripe_height).__str__() + " " + (
                            i * fun.stripe_height + fun.stripe_height).__str__())
                    print("band resample:" + gb_resample_start_y.__str__() + " " + gb_resample_end_y.__str__())

                    # 用于判断基准影像是否为空，对应于当基准影像的黑色部分超过条带高度时，会引发重采异常，导致多复制影像
                    res = np.count_nonzero(band_g)
                    if (res * 1.0) / (band_g.shape[0] * band_g.shape[1]) < 0.2:
                        print("base image is empty.")
                        resampled_band_b = np.zeros([fun.stripe_height, img_w_g], np.uint8)
                        img_parts.append(resampled_band_b)

                        t2 = time.time()
                        dt = t2 - t1
                        cost_time.append(dt)
                        print("cost time:" + dt.__str__())
                        continue

                    if fun.isCloudMode:
                        kps_gb_g, kps_gb_b, match_img = fun.alignRobust2BandsCloud(band_g, band_b)
                    else:
                        # 在基准影像不为空的情况下进行匹配，获得仿射矩阵，进行重采
                        kps_gb_g, kps_gb_b, match_img = fun.alignRobust2Bands(band_g, band_b)
                    if fun.isDebugMode:
                        cv2.imwrite(
                            "output/match_" + img_name1 + "_" + img_name2 + '_' + i.__str__().zfill(2) + ".jpg",
                            match_img)
                        fout = open("output/match_" + img_name1 + "_" + img_name2 + '_' + i.__str__().zfill(2) + ".txt",
                                    'w')
                        for kp1, kp2 in zip(kps_gb_g, kps_gb_b):
                            fout.write(
                                kp1[0].__str__() + "\t" + kp1[1].__str__() + "\t" + kp2[0].__str__() + "\t" + kp2[
                                    1].__str__() + "\n")
                        fout.close()

                    if kps_gb_g.__len__() < 3:
                        if affine_matrices_gb.__len__() == 0:
                            print("No affine matrix to use,copy stripe image to dst stripe.")
                            img_parts.append(band_g_resample)
                            if fun.isDebugMode:
                                cv2.imwrite("output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                                            band_b)
                            continue
                        else:
                            print("Number of match points is less than 3,"
                                  "can't estimate affine matrix.Use last affine matrix.")
                            affine1 = affine_matrices_gb[-1]
                    else:
                        affine1, mask = cv2.estimateAffine2D(np.array(kps_gb_b), np.array(kps_gb_g))
                        if affine1 is None:
                            if affine_matrices_gb.__len__() == 0:
                                print("estimated affine matrix is none and no affine matrix to use,"
                                      "copy stripe image to dst stripe.")
                                img_parts.append(band_g_resample)
                                if fun.isDebugMode:
                                    cv2.imwrite("output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                                                band_b)
                                continue
                            else:
                                print("Number of match points is less than 3,"
                                      "can't estimate affine matrix.Use last affine matrix.")
                                affine1 = affine_matrices_gb[-1]
                        else:
                            num1 = affine1[0][0]
                            num2 = affine1[1][1]
                            print(num1, num2)
                            if fun.affine_min < num1 < fun.affine_max and fun.affine_min < num2 < fun.affine_max:
                                affine_matrices_gb.append(affine1)
                            else:
                                print("Estimated affine matrix is wrong.Try to use last affine matrix.")
                                if affine_matrices_gb.__len__() == 0:
                                    print("No affine matrix to use,copy stripe image to dst stripe.")
                                    img_parts.append(band_g_resample)
                                    if fun.isDebugMode:
                                        cv2.imwrite(
                                            "output/align_" + img_name2 + "_" + (i + 1).__str__().zfill(2) + ".jpg",
                                            band_b)
                                    continue
                                else:
                                    print(
                                        "Success use last affine matrix.")
                                    affine1 = affine_matrices_gb[-1]
                    print(affine1)

                    # 重采
                    resampled_band_b = cv2.warpAffine(band_b_resample, affine1,
                                                      (band_g_resample.shape[1], band_g_resample.shape[0]))

                    # 添加不同条带到list
                    img_parts.append(resampled_band_b)

                    t2 = time.time()
                    dt = t2 - t1
                    cost_time.append(dt)
                    print("cost time:" + dt.__str__())

                # 剩余部分处理，保证重采后的图与原图大小相同
                print("\nDeal with residual part...")
                residual_part_g = np.zeros([img_h_g - fun.stripe_height * blocks, img_w_g], np.uint8)
                img_parts.append(residual_part_g)

                print("Total cost time:" + sum(cost_time).__str__() + " s")
                img_out = img_parts[0]
                print("\nMosaic images...")
                for i in range(1, img_parts.__len__()):
                    img_out = np.vstack((img_out, img_parts[i]))
                print("Save image...")
                cv2.imwrite(out_path, img_out)
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
