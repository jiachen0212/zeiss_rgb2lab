# coding=utf-8
import os
import json
import random
import xlrd
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils.util import calculate_Lab

def gamma(a):
    if a > 0.04045:
        a = np.power((a+0.055)/1.055, 2.4)
    else:
        a /= 12.92

    return a


def base_green_blue_check_red(data_lab, data_rgb):
    '''
    L* 体现亮度
    a* 越小则越绿, 对应green越小; a*越大则越红, 对应red越大
    b* 越小则越蓝 越大则越黄  [b*大则可能green blue都大; b*大则可能仅blue大]

    基于LAB值, 对比RGB值是否正确

    L = [9.5, 14.5]
    A = [-24, -15]
    B = [-2, 10]

    '''
    all_bad = []

    error_lab_rgb = open(r'./error_lab_rgb.txt', 'w')
    title1 = "L* a* similar, compare b*: \n"
    error_lab_rgb.write(title1)

    labs = []
    rgbs = []
    ks = []
    for k, v in data_lab.items():
        labs.append([float(a) for a in v])
        rgbs.append([float(a) for a in data_rgb[k]])
        ks.append(k)

    nums = len(labs)
    for i in range(nums):
        cur_lab = labs[i]
        cur_rgb = rgbs[i]
        tmp_lab = labs[:i] + labs[i + 1:]
        tmp_rgb = rgbs[:i] + rgbs[i + 1:]
        tmp_ks = ks[:i] + ks[i + 1:]
        for j, tmp_lab_ in enumerate(tmp_lab):
            # La值接近
            if (abs(tmp_lab_[0]-cur_lab[0]) <= 0.08) and (abs(tmp_lab_[1]-cur_lab[1]) <= 0.08):
                tmp_rgb_ = tmp_rgb[j]
                tmp_name = tmp_ks[j]
                if tmp_lab_[2] >= cur_lab[2]:
                    # b* 越大, 则green blue都大
                    if (tmp_rgb_[1] > cur_rgb[1]) and (tmp_rgb_[2] > cur_rgb[2]):
                        pass
                    else:
                        # line1 = "cur_lab: {},  cur_rgb: {}".format(cur_lab, cur_rgb)
                        # line2 = "similar_lab: {}, similar_rgb: {}".format(tmp_lab_, tmp_rgb_)
                        # print(line1, line2)
                        pass
                    # b*大则越黄, 不好量化更蓝是怎样的GB组合
                else:
                    # b* 更小则越蓝
                    if (tmp_rgb_[2] <= cur_rgb[2]) and (abs(tmp_rgb_[1] - cur_rgb[1]) <= 1):
                        line1 = "img_name: {}, cur_lab: {},  similar_img_name: {}, similar_lab: {}".format(ks[i], cur_lab, tmp_name, tmp_lab_)
                        line2 = "cur_rgb: {}, similar_rgb: {}".format(cur_rgb, tmp_rgb_)
                        info = line1 + line2
                        error_lab_rgb.write(info+'\n')
                        all_bad.extend([ks[i], tmp_name])
    error_lab_rgb.write('\n')
    title2 = "L* b* similar, compare a*: \n"
    error_lab_rgb.write(title2)
    # a* 越小则越绿, 对应green越小; a*越大则越红, 对应red越大
    for i in range(nums):
        cur_lab = labs[i]
        cur_rgb = rgbs[i]
        tmp_lab = labs[:i] + labs[i + 1:]
        tmp_rgb = rgbs[:i] + rgbs[i + 1:]
        tmp_ks = ks[:i] + ks[i + 1:]
        for j, tmp_lab_ in enumerate(tmp_lab):
            # Lb值接近
            if (abs(tmp_lab_[0]-cur_lab[0]) <= 0.08) and (abs(tmp_lab_[2]-cur_lab[2]) <= 0.08):
                tmp_rgb_ = tmp_rgb[j]
                tmp_name = tmp_ks[j]
                if tmp_lab_[1] >= cur_lab[1]:
                    # a更大则更红, R值更大
                    if (tmp_rgb_[0] <= cur_rgb[0]) and (abs(tmp_rgb_[1] - cur_rgb[1]) <= 1) and (abs(tmp_rgb_[2] - cur_rgb[2]) <= 1):
                        line1 = "img_name: {}, cur_lab: {},  similar_img_name: {}, similar_lab: {}".format(ks[i], cur_lab, tmp_name, tmp_lab_)
                        line2 = "cur_rgb: {}, similar_rgb: {}".format(cur_rgb, tmp_rgb_)
                        info = line1 + line2
                        error_lab_rgb.write(info + '\n')
                        all_bad.extend([ks[i], tmp_name])
                else:
                    # a更小, 则G更大更绿
                    if (tmp_rgb_[1] <= cur_rgb[1]) and (abs(tmp_rgb_[0] - cur_rgb[0]) <= 1) and (
                            abs(tmp_rgb_[2] - cur_rgb[2]) <= 1):
                        line1 = "img_name: {}, cur_lab: {},  similar_img_name: {}, similar_lab: {}".format(ks[i], cur_lab, tmp_name, tmp_lab_)
                        line2 = "cur_rgb: {}, similar_rgb: {}".format(cur_rgb, tmp_rgb_)
                        info = line1 + line2
                        error_lab_rgb.write(info + '\n')
                        all_bad.extend([ks[i], tmp_name])

    # slim_bad_pairs = []
    # for tmp in all_bad:
    #     if tmp not in slim_bad_pairs:
    #         slim_bad_pairs.append(tmp)
    # print(len(slim_bad_pairs))


def show_rgb_lab_distracte(data1_lab, data1_rgb):
    L,A,B = [],[],[]
    r,g,b = [], [], []
    grammed_r, grammed_g, grammed_b = [], [], []
    for k, v in data1_lab.items():
        L.append(float(v[0]))
        A.append(float(v[1]))
        B.append(float(v[2]))
        r.append(float(data1_rgb[k][0]))
        g.append(float(data1_rgb[k][1]))
        b.append(float(data1_rgb[k][2]))
        grammed_r.append(gamma(float(data1_rgb[k][0])))
        grammed_g.append(gamma(float(data1_rgb[k][1])))
        grammed_b.append(gamma(float(data1_rgb[k][2])))


    plt.hist(x=r, bins='auto', color='red', alpha=0.7, rwidth=0.85, label='r')
    plt.hist(x=g, bins='auto', color='green', alpha=0.7, rwidth=0.85, label='g')
    plt.hist(x=b, bins='auto', color='blue', alpha=0.7, rwidth=0.85, label='b')
    plt.grid(axis='y', alpha=0.75)
    plt.title('rgb')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(x=grammed_r, bins='auto', color='red', alpha=0.7, rwidth=0.85, label='r')
    plt.hist(x=grammed_g, bins='auto', color='green', alpha=0.7, rwidth=0.85, label='g')
    plt.hist(x=grammed_b, bins='auto', color='blue', alpha=0.7, rwidth=0.85, label='b')
    plt.grid(axis='y', alpha=0.75)
    plt.title('gammaed_rgb')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


    plt.hist(x=L, bins='auto', color='lightsalmon', alpha=0.7, rwidth=0.85, label='L')
    plt.hist(x=A, bins='auto', color='pink', alpha=0.7, rwidth=0.85, label='A')
    plt.hist(x=B, bins='auto', color='darkslateblue', alpha=0.7, rwidth=0.85, label='B')
    plt.grid(axis='y', alpha=0.75)
    plt.title('data1-LAB')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



def ABCD_light():
    dir_ = r'D:\work\project\卡尔蔡司膜色缺陷\阶段结论文档\不同光源功率值'
    colors = ['blue', 'black', 'red', 'yellow', 'pink']
    names = ['D50.txt', 'D55.txt', 'D65.txt', 'D75.txt']   # 'A.txt', 'B.txt', 'C.txt',
    for ind, txt in enumerate(names):
        curve = open(os.path.join(dir_, txt), 'r').readlines()
        curve = [float(a) for a in curve]
        plt.plot([i*5+380 for i in range(len(curve))], curve, color=colors[ind], label=txt[:-4])
    plt.legend()
    plt.grid()
    plt.show()


def show_green_lab_curve(colors):
    file = r'D:\work\project\卡尔蔡司膜色缺陷\阶段结论文档\0118zeiss_膜色缺陷对齐\2022-01-07 原始数据.xlsx'
    wb = xlrd.open_workbook(file)
    inds = [i for i in range(2, 19)]
    inds.remove(3)
    inds.remove(11)
    sheets = ["Sheet1+", "Sheet3+"] + ["Sheet{}".format(ind) for ind in inds] + ["Sheet11+"]
    print(len(sheets))
    all_lab_curve = dict()
    for sheet in sheets:
        key = sheet.strip('Shet+')
        all_lab_curve[key] = []
        data = wb.sheet_by_name(sheet)
        rows = data.nrows
        for i in range(1, rows):
            row_data = data.row_values(i)
            curve = row_data[15: 97]
            if len(curve) > 0:
                all_lab_curve[key].append(curve)
        print(sheet, len(all_lab_curve[key]))

    aa = [380+5*i for i in range(81)]
    for ind, dir_ in enumerate(list(all_lab_curve.keys())):
        i = 0
        for curve in all_lab_curve[dir_]:
            if i == 0:
                plt.plot(aa, curve, colors[ind], label=dir_)
                i += 1
            else:
                plt.plot(aa, curve, colors[ind])
    plt.legend()
    plt.grid()
    plt.show()


def show_blue_lab_curve(colors):
    file = r'D:\work\project\卡尔蔡司膜色缺陷\1209\膜色识别~测试 12-09th.xlsx'
    wb = xlrd.open_workbook(file)
    inds = [i for i in range(16, 22)]
    inds.remove(17)
    sheets = ["Sheet17+"] + ["Sheet{}".format(ind) for ind in inds]
    all_lab_curve = dict()
    for sheet in sheets:
        key = sheet.strip('Shet+')
        all_lab_curve[key] = []
        data = wb.sheet_by_name(sheet)
        rows = data.nrows
        for i in range(1, rows):
            row_data = data.row_values(i)
            curve = row_data[14: 97]
            if len(curve) > 0:
                all_lab_curve[key].append(curve)
        print(sheet, len(all_lab_curve[key]))

    aa = [380+5*i for i in range(81)]
    for ind, dir_ in enumerate(list(all_lab_curve.keys())):
        i = 0
        for curve in all_lab_curve[dir_]:
            if i == 0:
                plt.plot(aa, curve, colors[ind], label=dir_)
                i += 1
            else:
                plt.plot(aa, curve, colors[ind])
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\22.01.7data\0107lab.json', 'r'))
    RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\22.01.7data\0107all_erode_dilate.json', 'r'))
    test_data_lab_gt = json.load(open(r'./0107_test_gt_lab.json', 'r'))
    all_lab = dict()
    for k, v in LAB.items():
        all_lab[k] = v
    for k, v in test_data_lab_gt.items():
        all_lab[k] = v

    data = json.dumps(all_lab)
    with open(r'./0107_train_test_lab.json', 'w') as js_file:
        js_file.write(data)

    assert len(all_lab) == len(RGB)

    # dir3随机留一半测一半
    dir3_lab = dict()
    inds = [i for i in range(1, 21)]
    random.shuffle(inds)
    test_dir3 = inds[:10]
    train_dir3 = inds[10:]

    # for k, v in all_lab.items():
    #     for ind in train_dir3:
    #         if k == "3_{}".format(ind):
    #             dir3_lab[k] = v
    # LAB.update(dir3_lab)

    # 留下7 10
    # dir7_lab = dict()
    # for k, v in all_lab.items():
    #     if k.split("_")[0] == "3":
    #         dir3_lab[k] = v
    # LAB.update(dir3_lab)
    # for k in list(LAB.keys()):
    #     if k.split("_")[0] == "7":
    #         dir7_lab[k] = LAB[k]
    #         del LAB[k]
    data = json.dumps(LAB)
    with open(r'./0119_train_lab.json', 'w') as js_file:
        js_file.write(data)
    # test_data_lab_gt.update(dir7_lab)
    # data = json.dumps(test_data_lab_gt)
    # with open(r'./0107_test_gt_lab1.json', 'w') as js_file:
    #     js_file.write(data)


    # base_green_blue_check_red(all_lab, RGB)

    # show_rgb_lab_distracte(all_lab, RGB)

    # compare ABCD light
    # ABCD_light()

    # show 0107_data lab_curve
    colors = ['pink', 'black', 'darkslateblue', 'green','blue', 'yellow', 'purple', 'green', 'cornflowerblue', 'red', 'turquoise'
              ,'dodgerblue', 'gray', 'hotpink', 'lavenderblush', 'darkseagreen', 'ghostwhite', 'firebrick']

    show_green_lab_curve(colors)
    # show_blue_lab_curve(colors)

    # lab_curve2LAb_value
    curve = [6.09, 4.1, 2.42, 1.41, 0.83, 0.59, 0.41, 0.27, 0.21, 0.18, 0.42, 0.74, 0.81, 0.85, 0.94, 1.28, 1.49, 1.61, 1.7, 1.8, 1.88, 2.05, 2.31, 2.45, 2.45, 2.42, 2.28, 2.26, 2.41, 2.51, 2.54, 2.4, 2.18, 2.04, 1.78, 1.65, 1.68, 1.58, 1.48, 1.38, 1.1, 0.88, 0.63, 0.46, 0.42, 0.42, 0.28, 0.3, 0.18, 0.18, 0.08, 0.03, 0.38, 0.46, 0.36, 0.67, 0.86, 0.96, 1.15, 1.56, 1.8, 1.95, 2.26, 2.46, 2.88, 3.2, 3.82, 4.18, 4.67, 5.25, 5.63, 6.1, 6.5, 6.93, 7.37, 7.89, 8.34, 8.67, 9.11, 9.79, 10.54]
    L, a, b = calculate_Lab(curve)
    print(L, a, b)




