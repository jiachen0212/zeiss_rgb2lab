# coding=utf-8
import json
import numpy as np
import os

tmp_save_dir = r'D:\work\project\卡尔蔡司膜色缺陷\data'


data1_lab = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924green_lab.json', 'r'))
data1_rgb = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924green_rgb.json', 'r'))
# data1_lab = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_lab.json', 'r'))
# data1_rgb = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_rgb.json', 'r'))

ks = []
labs = []
rgbs = []
for k, v in data1_lab.items():
    ks.append(k)
    labs.append([float(a) for a in v])
    rgbs.append([float(a) for a in data1_rgb[k]])


# ff = data_diff_info = open(os.path.join(tmp_save_dir, r'./data1.diff.txt'), 'w')
ff = data_diff_info = open(os.path.join(tmp_save_dir, r'./0924green.diff.txt'), 'w')

nums = len(labs)
# rgb接近的, 查看lab diff情况
for i in range(nums):
    cur_lab = labs[i]
    cur_rgb = rgbs[i]
    tmp_lab = labs[:i] + labs[i+1:]
    tmp_rgb = rgbs[:i] + rgbs[i+1:]
    diff_rgb = [(abs(tmp_rgb[j][0] - cur_rgb[0]))+(abs(tmp_rgb[j][1] - cur_rgb[1]))+(abs(tmp_rgb[j][2] - cur_rgb[2])) for j in range(nums-1)]
    min_diff_rgb_index = diff_rgb.index(min(diff_rgb))
    lab_rgb_ = [tmp_rgb[min_diff_rgb_index][b] - cur_rgb[b] for b in range(3)]
    # print("minest rgb diff : {}".format(lab_rgb_))
    min_diff_rgb_index_lab = tmp_lab[min_diff_rgb_index]
    min_index_lab_diff = [min_diff_rgb_index_lab[a] - cur_lab[a] for a in range(3)]
    # print("minest rgb's lab diff : {}".format(min_index_lab_diff))
    # print('\n')


# lab接近的, 查看rgb diff情况
for i in range(nums):
    cur_lab = labs[i]
    cur_rgb = rgbs[i]
    tmp_lab = labs[:i] + labs[i+1:]
    tmp_rgb = rgbs[:i] + rgbs[i+1:]
    diff_lab = [(abs(tmp_lab[j][0] - cur_lab[0]))+(abs(tmp_lab[j][1] - cur_lab[1]))+(abs(tmp_lab[j][2] - cur_lab[2])) for j in range(nums-1)]
    min_diff_lab_index = diff_lab.index(min(diff_lab))
    lab_lab_ = [tmp_lab[min_diff_lab_index][b] - cur_lab[b] for b in range(3)]
    # print("minest lab diff : {}".format(lab_lab_))
    min_diff_lab_index_rgb = tmp_rgb[min_diff_lab_index]
    min_index_rgb_diff = [min_diff_lab_index_rgb[a] - cur_rgb[a] for a in range(3)]
    # print("minest lab's rgb diff : {}".format(min_index_rgb_diff))
    # print('\n')

    # for lab_diff__ in lab_lab_:
    #     if abs(lab_diff__) > 0.2:
    #         a = "lab diff: {}".format(lab_lab_)
    #         ff.write(a + '\n')
    #         b = "rgb diff: {}".format(min_index_rgb_diff)
    #         ff.write(b + '\n')
    #         print(a)
    #         print(b)
    #         break
    # print('--------------')
    # ff.write('\n')


    # lab_lab_abs = [abs(a) for a in lab_lab_]
    # flag = (np.array(lab_lab_abs) < 0.2).all()
    # if flag:
    #     a = "all lab diff in 0.2: {}".format(lab_lab_)
    #     ff.write(a + '\n')
    #     b = "rgb diff: {}".format(min_index_rgb_diff)
    #     ff.write(b + '\n')
    #     print(a)
    #     print(b)
    # print('--------------')
    # ff.write('\n')

    lab_lab_abs = [abs(a) for a in lab_lab_]
    flag = (np.array(lab_lab_abs) > 0.5).any()
    if flag:
        a = "any lab diff larger than 0.5: {}".format(lab_lab_)
        ff.write(a + '\n')
        b = "rgb diff: {}".format(min_index_rgb_diff)
        ff.write(b + '\n')
        print(a)
        print(b)
    print('--------------')
    ff.write('\n')
