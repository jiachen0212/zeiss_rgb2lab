# coding=utf-8
'''
rgb lab 一致性检查.. data正确性快速验证

'''
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def fun(data1_lab, data1_rgb):
    tmp_save_dir = r'D:\work\project\卡尔蔡司膜色缺陷\data'

    ks = []
    labs = []
    rgbs = []
    for k, v in data1_lab.items():
        ks.append(k)
        labs.append([float(a) for a in v])
        rgbs.append([float(a) for a in data1_rgb[k]])


    # ff = open(os.path.join(tmp_save_dir, r'./data1.diff.txt'), 'w')
    ff = open(os.path.join(tmp_save_dir, r'./0924green.diff.txt'), 'w')

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



def base_red_blue_compare_green(data_lab, data_rgb, save_dir):
    '''
    check 绿膜数据的lab rgb 一致性
    '''

    ff = open(os.path.join(save_dir, r'green_data_lab_rgb_check.txt'), 'w')
    aa = open(os.path.join(save_dir, r'green_data_lab_rgb.txt'), 'w')
    labs = []
    rgbs = []
    ks = []
    for k, v in data_lab.items():
        labs.append([float(a) for a in v])
        rgbs.append([float(a) for a in data_rgb[k]])
        ks.append(k)

    a, b = 0, 0
    bad_pair = []
    nums = len(labs)
    for i in range(nums):
        cur_lab = labs[i]
        cur_rgb = rgbs[i]
        tmp_lab = labs[:i] + labs[i + 1:]
        tmp_rgb = rgbs[:i] + rgbs[i + 1:]
        tmp_ks = ks[:i] + ks[i + 1:]
        # a, b = 0, 0
        for j, tmp_rgb_ in enumerate(tmp_rgb):
            # rgb的rb值差值<1
            if (abs(tmp_rgb_[2]-cur_rgb[2]) < 1) and (abs(tmp_rgb_[0]-cur_rgb[0]) < 1):
                a += 1
                tmp_lab_ = tmp_lab[j]
                if ((tmp_rgb_[1] > cur_rgb[1]) and (tmp_lab_[1] < cur_lab[1]) or ((tmp_rgb_[1] < cur_rgb[1]) and (tmp_lab_[1] > cur_lab[1]))):
                    line1 = "cur_rgb: {}, cur_lab: {}".format(cur_rgb, cur_lab)
                    # print(line1)
                    line2 = "tmp_rgb: {}, tmp_lab: {}".format(tmp_rgb_, tmp_lab_)
                    # print(line2)
                    # print('\n')
                    # ff.write(line1 + '\n')
                    # ff.write(line2 + '\n')
                    # ff.write('\n')

                    b += 1
                else:
                    bad_pair.append([ks[i], tmp_ks[j]])
    print("all blue similarity: {}, matched green: {}".format(a, b))
    print(len(bad_pair))
    slim_bad_pair = []
    for i in range(len(bad_pair)):
        slim_bad_pair.append(bad_pair[i])
        for j in range(i, len(bad_pair)):
            if bad_pair[i] == [bad_pair[j][1], bad_pair[j][0]]:
                slim_bad_pair.remove(bad_pair[i])
    print(len(slim_bad_pair))
    # 输出不单调的lab和rgb
    for k in slim_bad_pair:
        line1 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[0]), data_lab[k[0]], data_rgb[k[0]])
        line2 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[1]), data_lab[k[1]], data_rgb[k[1]])
        # print(line1)
        # print(line2)
        # print('\n')
        aa.write(line1 + '\n')
        aa.write(line2 + '\n')
        aa.write('\n')

    return slim_bad_pair


def change_dir(k):
    return str(int(k.split('_')[0]))+'_'+k.split('_')[1]


def base_green_red_compare_blue(data_lab, data_rgb, save_dir):
    '''
    check 蓝膜数据的lab rgb 一致性
    '''

    # ff = open(os.path.join(save_dir, r'blue_data_lab_rgb_check.txt'), 'w')
    aa = open(os.path.join(save_dir, r'blue_data_lab_rgb.txt'), 'w')
    labs = []
    rgbs = []
    ks = []
    for k, v in data_lab.items():
        labs.append([float(a) for a in v])
        rgbs.append([float(a) for a in data_rgb[k]])
        # ks.append(change_dir(k))
        ks.append(k)

    a, b = 0, 0
    nums = len(labs)
    bad_pair = []
    for i in range(nums):
        cur_lab = labs[i]
        cur_rgb = rgbs[i]
        tmp_lab = labs[:i] + labs[i + 1:]
        tmp_rgb = rgbs[:i] + rgbs[i + 1:]
        tmp_ks = ks[:i] + ks[i+1:]
        # a, b = 0, 0
        for j, tmp_rgb_ in enumerate(tmp_rgb):
            # rgb的rg值差值<1
            if (abs(tmp_rgb_[1]-cur_rgb[1]) < 1) and (abs(tmp_rgb_[0]-cur_rgb[0]) < 1):
                a += 1
                tmp_lab_ = tmp_lab[j]
                if ((tmp_rgb_[2] > cur_rgb[2]) and (tmp_lab_[2] < cur_lab[2]) or ((tmp_rgb_[2] < cur_rgb[2]) and (tmp_lab_[2] > cur_lab[2]))):
                    line1 = "cur_rgb: {}, cur_lab: {}".format(cur_rgb, cur_lab)
                    # print(line1)
                    line2 = "tmp_rgb: {}, tmp_lab: {}".format(tmp_rgb_, tmp_lab_)
                    # print(line2)
                    # print('\n')
                    # ff.write(line1+'\n')
                    # ff.write(line2 + '\n')
                    # ff.write('\n')
                    b += 1
                else:
                    bad_pair.append([ks[i], tmp_ks[j]])
    print("all green similarity: {}, matched blue: {}".format(a, b))
    print(len(bad_pair))
    slim_bad_pair = []
    for i in range(len(bad_pair)):
        slim_bad_pair.append(bad_pair[i])
        for j in range(i, len(bad_pair)):
            if bad_pair[i] == [bad_pair[j][1], bad_pair[j][0]]:
                slim_bad_pair.remove(bad_pair[i])
    print(len(slim_bad_pair))
    # 输出不单调的lab和rgb
    for k in slim_bad_pair:
        line1 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[0]), data_lab[k[0]], data_rgb[k[0]])
        line2 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[1]), data_lab[k[1]], data_rgb[k[1]])
        # print(line1)
        # print(line2)
        # print('\n')
        aa.write(line1+'\n')
        aa.write(line2 + '\n')
        aa.write('\n')

    return slim_bad_pair



def base_green_blue_check_red(data_lab, data_rgb, save_dir):
    '''
    RGB中gb相近, r值越大, LAB的A值越小
    '''

    aa = open(os.path.join(save_dir, r'RG相近_lab_a值检查.txt'), 'w')
    labs = []
    rgbs = []
    ks = []
    for k, v in data_lab.items():
        labs.append([float(a) for a in v])
        rgbs.append([float(a) for a in data_rgb[k]])
        ks.append(k)

    a, b = 0, 0
    bad_pair = []
    nums = len(labs)
    for i in range(nums):
        cur_lab = labs[i]
        cur_rgb = rgbs[i]
        tmp_lab = labs[:i] + labs[i + 1:]
        tmp_rgb = rgbs[:i] + rgbs[i + 1:]
        tmp_ks = ks[:i] + ks[i + 1:]
        for j, tmp_rgb_ in enumerate(tmp_rgb):
            # rgb的gb值差值<1
            if (abs(tmp_rgb_[2]-cur_rgb[2]) <= 0.5) and (abs(tmp_rgb_[1]-cur_rgb[1]) <= 0.5):
                a += 1
                tmp_lab_ = tmp_lab[j]
                if ((tmp_rgb_[0] > cur_rgb[0]) and (tmp_lab_[0] > cur_lab[0]) or ((tmp_rgb_[0] <= cur_rgb[0]) and (tmp_lab_[0] <= cur_lab[0]))):
                    b += 1
                else:
                    bad_pair.append([ks[i], tmp_ks[j]])
    slim_bad_pair = []
    for i in range(len(bad_pair)):
        slim_bad_pair.append(bad_pair[i])
        for j in range(i, len(bad_pair)):
            if bad_pair[i] == [bad_pair[j][1], bad_pair[j][0]]:
                slim_bad_pair.remove(bad_pair[i])
    # 输出不单调的lab和rgb
    for k in slim_bad_pair:
        line1 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[0]), data_lab[k[0]], data_rgb[k[0]])
        line2 = "dir_name1: {}, lab: {}, rgb: {}".format(change_dir(k[1]), data_lab[k[1]], data_rgb[k[1]])
        # print(line1)
        # print(line2)
        # print('\n')
        aa.write(line1 + '\n')
        aa.write(line2 + '\n')
        aa.write('\n')

    return slim_bad_pair




def json2csv(data1_lab, data1_rgb):
    data1 = pd.DataFrame()
    ks = []
    labs = []
    rgbs = []
    for k, v in data1_lab.items():
        ks.append(k)
        labs.append(v)
        rgbs.append(data1_rgb[k])
    # 数据写入csv
    data1['dir_name'] = ks
    data1['lab'] = labs
    data1['rgb'] = rgbs
    data1.to_csv(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_lab_rgb.csv', index=False)


def gamma(a):
    if a > 0.04045:
        a = np.power((a+0.055)/1.055, 2.4)
    else:
        a /= 12.92

    return a


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





if __name__ == "__main__":

    # json2csv(data1_lab, data1_rgb)

    # RGB中, RG接近, B越大b越小; GB接近, R越大a越大; RB接近, G越大b越小..
    data1_lab = json.load(open(r'./1209_green_test_lab.json', 'r'))
    data1_rgb = json.load(open(r'./1209_test_rgb.json', 'r'))

    save_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209\rgb_lab一致性检查'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 蓝膜数据
    # data1_lab = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_lab.json', 'r'))
    # data1_rgb = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_rgb.json', 'r'))

    # fun(data1_lab, data1_rgb)

    # bad_green = base_red_blue_compare_green(data1_lab, data1_rgb, save_dir)
    # bad_blue = base_green_red_compare_blue(data1_lab, data1_rgb, save_dir)
    bad_red = base_green_blue_check_red(data1_lab, data1_rgb, save_dir)

    # all_bad = []
    # for bad_list in [bad_green, bad_blue, bad_red]:
    #     for a in bad_list:
    #         all_bad.extend(a)
    #
    # set_all_bad = list(set(all_bad))
    # print(len(set_all_bad))
    # print(set_all_bad)

    # show_rgb_lab_distracte(data1_lab, data1_rgb)

