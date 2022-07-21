# coding=utf-8
import cv2
import os
from shutil import copyfile

copy_path = r'C:\Users\15974\Desktop\1'
def rename_copy(copy_path):
    img_dir = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\膜色缺陷\mose0712'
    dirs = [os.path.join(img_dir, a) for a in os.listdir(img_dir) if 'RGB' in a]

    for dir_ in dirs:
        pre = os.path.basename(dir_).split('-')[0]
        ims = os.listdir(dir_)
        for im_name in ims:
            # new_name = os.path.join(dir_, "{}_{}".format(pre, im_name))
            # os.rename(os.path.join(dir_, im_name), new_name)
            copyfile(os.path.join(dir_, im_name), os.path.join(copy_path, im_name))

rename_copy(copy_path)