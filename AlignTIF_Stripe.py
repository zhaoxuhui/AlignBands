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
        total_cost_time = []

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

                # 转8bit用于匹配处理，直接整体拉伸，不要每条带单独拉伸
                # 单独拉伸会造成一是匹配的特征点非常多，二是在暗部较多的条带中，可能会导致过曝，丢失细节
                # 同时注意图像中缺失的黑块处理，否则可能会影响整体拉伸效果，暂时用128填充
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
                    band_b_resample = band_b_10[gb_resample_start_y:gb_resample_end_y, :]
                    band_g_resample = band_g_10[i * fun.stripe_height:i * fun.stripe_height + fun.stripe_height, :]

                    print("band base:" + (i * fun.stripe_height).__str__() + " " + (
                            i * fun.stripe_height + fun.stripe_height).__str__())
                    print("band resample:" + gb_resample_start_y.__str__() + " " + gb_resample_end_y.__str__())

                    # 用于判断基准影像是否为空，对应于当基准影像的黑色部分超过条带高度时，会引发重采异常，导致多复制影像
                    res = np.count_nonzero(band_g_resample)
                    if (res * 1.0) / (band_g_resample.shape[0] * band_g_resample.shape[1]) < 0.2:
                        print("base image is empty.")
                        resampled_band_b = np.zeros([fun.stripe_height, img_w_g], np.uint16)
                        img_parts.append(resampled_band_b)

                        t2 = time.time()
                        dt = t2 - t1
                        cost_time.append(dt)
                        print("cost time:" + dt.__str__())
                        continue

                    # 最新发现的问题
                    # 对于某些影像而言，必须每个条带单独拉伸
                    # 因为对于影像中既有非常亮又有非常暗的区域，整体拉伸会导致整个影像过曝或过暗，从而导致特征丢失
                    # 而因拉伸导致的特征点过多问题其实不算问题，可以通过动态阈值的办法缓解或解决
                    # 现在更关键的问题是提不到特征点，匹配点对太少，从而导致仿射矩阵计算失败
                    # 相比于影像无法对准，多一点计算时间也不算什么问题，毕竟更关注的是处理后的结果好坏
                    # 在基准影像不为空的情况下进行匹配，获得仿射矩阵，进行重采
                    # 这里给出一个初步解决方案：
                    # 读取原始影像，转8位，找最大最小值，作差，若大于某阈值(如100)，说明影像灰度变换范围大，不宜做整体拉伸，适合做分条带拉伸
                    # 若小于阈值，则说明灰度分布比较集中，拉伸会比较有效果，这时就不需要再分条带拉伸了
                    if isStripeStretch:
                        if fun.isCloudMode:
                            kps_gb_g, kps_gb_b = fun.alignRobust2BandsTIFWithStretchCloud(band_g, band_b, it + 1, i + 1)
                        else:
                            kps_gb_g, kps_gb_b = fun.alignRobust2BandsTIFWithStretch(band_g, band_b, it + 1, i + 1)
                    else:
                        if fun.isCloudMode:
                            kps_gb_g, kps_gb_b = fun.alignRobust2BandsTIFCloud(band_g, band_b, it + 1, i + 1)
                        else:
                            kps_gb_g, kps_gb_b = fun.alignRobust2BandsTIF(band_g, band_b, it + 1, i + 1)

                    # 当匹配点对小于3，无法构建仿射矩阵
                    if kps_gb_g.__len__() < 3:
                        # 判断上一个条带是否成功生成仿射矩阵，如果是的话，使用上一个矩阵，否的话直接复制影像结束本次循环
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
                    # 匹配点对大于3，继续判断能否生成仿射矩阵以及生成的是否正确
                    else:
                        affine1, mask = cv2.estimateAffine2D(np.array(kps_gb_b), np.array(kps_gb_g))
                        # 如果生成失败，判断上一个条带是否成功生成仿射矩阵，如果是的话，使用上一个矩阵，否的话直接复制影像结束本次循环
                        if affine1 is None:
                            if affine_matrices_gb.__len__() == 0:
                                print("estimated affine matrix is none and no matrix to use, "
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
                        # affine matrix不为空，再判断是否正确
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
                                    print("Success use last affine matrix.")
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
                residual_part_g = np.zeros([img_h_g - fun.stripe_height * blocks, img_w_g], np.uint16)
                img_parts.append(residual_part_g)

                total_cost_time.append(sum(cost_time))
                print("Total cost time:" + sum(cost_time).__str__() + " s")
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
            print("All bands total cost time:" + sum(total_cost_time).__str__() + " s")
            os.system('pause')
        else:
            os.system('pause')
    else:
        print("Input 'yourExeName.exe help' to get help information.")
        os.system('pause')
