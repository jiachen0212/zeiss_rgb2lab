# coding=utf-8
import os
import json


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
                    # b*大, 不好量化更蓝是怎样的GB组合
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

    slim_bad_pairs = []
    for tmp in all_bad:
        if tmp not in slim_bad_pairs:
            slim_bad_pairs.append(tmp)
    print(len(slim_bad_pairs))

if __name__ == "__main__":
    LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\1209_green_lab.json', 'r'))
    RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\1209_all_green_rgb.json', 'r'))
    test_data_lab_gt = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\1209\1216zeiss对齐材料\1209_test_lab_gt.json', 'r'))
    all_lab = dict()
    for k, v in LAB.items():
        all_lab[k] = v
    for k, v in test_data_lab_gt.items():
        all_lab[k] = v

    base_green_blue_check_red(all_lab, RGB)