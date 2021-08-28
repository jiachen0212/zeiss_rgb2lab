# coding=utf-8
'''
直接rgb值计算M矩阵, 得到lab值.

'''



import json
import numpy as np
from numpy.linalg import solve
from numpy.linalg import lstsq

def fun(xyz_3, rgb_9):
    a = np.mat(rgb_9)
    b = np.mat(xyz_3).T
    weights_9 = lstsq(a, b)

    return weights_9

def xyz2lab(x, y, z):
    x /= 94.81211415
    y /= 100
    z /= 107.3369399
    if y > 0.008856:
        fy = np.power(y, 1/3)
    else:
        fy = 7.787 * y + 16 / 116.0
    if x > 0.008856:
        fx = np.power(x, 1/3)
    else:
        fx = 7.787 * x + 16 / 116.0
    if z > 0.008856:
        fz = np.power(z, 1/3)
    else:
        fz = 7.787 * z + 16 / 116.0
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return [l, a, b]


def lab2xyz(l,a,b, ff):
    # print("l: {}, a: {}, b: {}".format(l, a, b))
    ff.write(str(l)+','+str(a)+','+str(b)+',')
    fy = (l+16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    if np.power(fy, 3) > 0.008856:
        y = np.power(fy, 3)
    else:
        y = (fy - 16 / 116.0) / 7.787

    if np.power(fx, 3) > 0.008856:
        x = np.power(fx, 3)
    else:
        x = (fx - 16 / 116.0) / 7.787

    if np.power(fz, 3) > 0.008856:
        z = np.power(fz, 3)
    else:
        z = (fz - 16 / 116.0) / 7.787
    x *= 94.81211415
    y *= 100
    z *= 107.3369399
    # print("x: {}, y: {}, z: {}".format(x, y, z))
    ff.write(str(x) + ',' + str(y) + ',' + str(z) + '\n')

    return [x,y,z]


def rgb2lab(ff, key, rgb, gt):
    weight = np.load('./weight.npy')
    x, y, z = 0, 0, 0
    for i in range(3):
        x += rgb[i]*weight[i][0]
        y += rgb[i]*weight[i+3][0]
        z += rgb[i]*weight[i+6][0]
    # print("x: {}, y: {}, z: {}".format(x,y,z))
    lab_pred = xyz2lab(x, y, z)
    diff = [gt[i] - lab_pred[i] for i in range(3)]
    out_info = "number: {}, lab_diff: {}".format(key, diff)
    # print(out_info)
    ff.write(out_info+'\n')
    diff_abs = [abs(a) for a in diff]
    if sum(diff_abs) <= 1.5:
        c = 1
    else:
        c = 0
    return c
    # print("lab: {}".format(lab_pred))



def get_and_check_weight(dir_index):
    print('-'*10, "dir: {}".format(dir_index))



    # data before 0812
    rgb_data = json.load(open(r'./all_col3.json', 'r'))
    lab_data = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\all_lab_value.json', 'r'))

    # 0812 data
    # rgb_data = json.load(open(r'./rgb6_0812_key1_8.json', 'r'))
    # lab_data = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\0812lab_value.json', 'r'))

    # assert len(lab_data) == len(rgb_data)

    # get weight
    xyz_3 = []
    rgb_9 = []
    ff = open(r'./rgb_lab_xyz.txt', 'w')
    for key, rgb in rgb_data.items():
        # rgb = [float(a) for a in rgb]
        # rgb = [(rgb[i]+rgb[i+3])/2 for i in range(3)]
        # 只算单个文件夹下的weight
        if key.split('_')[0] == str(dir_index):
            # print(key)
            # print("r g b: {}".format(rgb))
            ff.write(''.join(a+',' for a in rgb))
            rgb = [float(a) / 255 for a in rgb]
            real_lab = lab_data[key]
            xyz = lab2xyz(real_lab[0], real_lab[1], real_lab[2], ff)
            xyz_3.extend(xyz)
            rgb_9.extend([rgb + [0] * 6, 3 * [0] + rgb + 3 * [0], [0] * 6 + rgb])
            # print("rgb_9: {}".format(rgb_9))
            # print("xyz_3: {}".format(xyz_3))
    weight = fun(xyz_3, rgb_9)
    # print(weight)
    np.save('./weight.npy', weight[0])

    good_count = 0
    fff = open(r'./result.txt', 'w')
    c = 0
    for key, rgb in rgb_data.items():
        # rgb = [float(a) for a in rgb]
        # rgb = [(rgb[i]+rgb[i+3])/2 for i in range(3)]
        if key.split('_')[0] == str(dir_index):
            c += 1
            rgb = [float(a) / 255 for a in rgb]
            good_count += rgb2lab(fff, key, rgb, gt=lab_data[key])
    print("all data size: {}".format(c))
    print("diff in [-0.5, 0.5] data size: {}".format(good_count))





def process_data():
    data08121 = json.load(open(r'./all_col6_0817.json', 'r'))
    modified_key_6rgb = dict()
    for k, v in data08121.items():
        modified_key_6rgb[k[1:]] = v
    data = json.dumps(modified_key_6rgb)
    with open('./rgb6_0812_key1_8.json', 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":
    # process_data()
    inds = [i for i in range(1, 7)]
    for ind in inds:
        get_and_check_weight(ind)







