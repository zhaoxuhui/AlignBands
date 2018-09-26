# coding=utf-8
import os.path
import sys
import shutil


def findAllFiles(root_dir, filter):
    """
    读取root_dir目录下指定类型文件的路径，返回一个list

    :param root_dir: 文件存放的目录
    :return: 返回两个list，paths为文件的绝对路径，names为文件名
    """

    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    # 遍历
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        print(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    return paths, names


def copyFiles(srcPaths, srcFiles, dstDir):
    for i in range(srcFiles.__len__()):
        shutil.copy(srcPaths[i] + srcFiles[i], dstDir + os.path.sep + srcFiles[i])
        print("Copied " + (i + 1).__str__() + "/" + srcFiles.__len__().__str__())


if sys.argv.__len__() == 2 and sys.argv[1] == "help":
    print("用于批量搜索制定文件并拷贝至指定位置\n")
    print("脚本启动命令格式：")
    print("scriptname.py:[search_dir] [fileType] [dst_dir]")
    print("\n函数帮助:")
    exec ("help(findAllFiles)")
    exec ("help(copyFiles)")
elif sys.argv.__len__() == 4:
    search_dir = sys.argv[1]
    fileType = sys.argv[2]
    dst_dir = sys.argv[3]
    paths, names = findAllFiles(search_dir, fileType)
    if sys.version_info.major < 3:
        flag = raw_input("\nStart copy?y/n\n")
    else:
        flag = input("\nStart copy?y/n\n")
    if flag == 'y':
        copyFiles(paths, names, dst_dir)
    else:
        exit()
else:
    print("Input \"scriptname.py help\" for help information.")
