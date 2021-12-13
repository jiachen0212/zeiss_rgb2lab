# coding=utf-8
'''
统计lab值分布情况 和 对应的g/b

'''



# import json
# import pandas as pd
#
# a = json.load(open('./1118_test_rgb.json', 'r'))
# df = pd.DataFrame()
# ks, vs = [], []
# for k, v in a.items():
#     ks.append(k)
#     vs.append(v[:-1])
# df['img_name'] = ks
# df['RGB'] = vs
#
# df.to_csv(r'./test_data_rgb.csv')
#
#
#
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gamma(a):
    if a > 0.04045:
        a = np.power((a+0.055)/1.055, 2.4)
    else:
        a /= 12.92

    return a

colors = ['aliceblue', 'antiquewhite', 'aqua', 'black', 'red', 'darkslateblue', 'turquoise',
              'blue', 'yellow', 'black', 'turquoise', 'green', 'cornflowerblue', 'purple', 'pink', 'green', 'cornflowerblue', 'purple', 'turquoise', 'green', 'cornflowerblue', 'purple', 'turquoise']
# 1209
# dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
# rgb_jss = os.listdir(dir)
# js = [a for a in rgb_jss if '.json' in a]
# plt.title("green")
# for i in range(1, 16):
#     js_file = r'D:\work\project\卡尔蔡司膜色缺陷\1209\dir_{}_rgb.json'.format(i)
#     js_data = json.load(open(os.path.join(dir, js_file), 'r'))
#     assert len(js_data) == 20
#     index = 0
#     for k, v in js_data.items():
#         if index == 0:
#             plt.plot([0,1,2], [gamma(float(a) / 255) for a in v], color=colors[i], label="dir_{}".format(i))
#         else:
#             plt.plot([0, 1, 2],[gamma(float(a) / 255) for a in v], color=colors[i])
#         index += 1
# plt.legend()
# plt.grid()
# plt.show()

base_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
rgb_jss = os.listdir(base_dir)
js = [a for a in rgb_jss if '.json' in a]

# for i3 in range(1, 16):
#     for i in range(1, 16):
#         js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
#         js_data = json.load(open(js_file, 'r'))
#         assert len(js_data) == 20
#         for k, v in js_data.items():
#             if i == i3:
#                 plt.plot([0, 1, 2],[gamma(float(a) / 255) for a in v], color='red')
#             else:
#                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
#     plt.title("dir{}_and_others".format(i3))
#     plt.legend()
#     plt.grid()
#     plt.savefig('./文件夹颜色差异/{}.png'.format(i3))
#     # plt.show()
#     plt.close()

dirs = [i for i in range(1, 16)]
green_test = json.load(open(r'./1209_test_rgb.json', 'r'))
for i in dirs:
    js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
    js_data = json.load(open(js_file, 'r'))
    assert len(js_data) == 20
    index = 0
    for k, v in js_data.items():
        if k in green_test:
            plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
        elif k in ['13_18', '11_4', '2_11', '13_7', '13_20']:
            plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='blue')
        # else:
        #     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
        index += 1
plt.legend()
plt.grid()
# plt.savefig('./文件夹颜色差异/each_dir/{}.png'.format(i))
plt.show()
# plt.close()


# for i in range(16,22):
#     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
#     js_data = json.load(open(js_file, 'r'))
#     assert len(js_data) == 20
#     keys = list(js_data.keys())
#     for test_index in range(20):
#         if i not in [2, 7]:
#             dir_ = os.path.join(r'D:\work\project\卡尔蔡司膜色缺陷\文件夹颜色差异\each_dir', str(i))
#             if not os.path.exists(dir_):
#                 os.makedirs(dir_)
#             for index, key in enumerate(keys):
#                 if index == test_index:
#                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js_data[key]], color='red',
#                              label="test_im: {}".format(key))
#                 else:
#                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js_data[key]], color='cornflowerblue')
#             plt.legend()
#             plt.grid()
#             plt.show()
#             # plt.savefig(os.path.join(dir_, '{}.png'.format(test_index)))
#             # plt.close()


# green
# test_ims = {"1": [19,20,2,4], "3": [2,3,10,14], "4": [8,9,19,20], "5": [2,20,5,11], "6": [4,5,18,20], "8": [1,4, 9, 15], "9": [1, 12, 10, 14], "10": [1, 2, 3, 12] , "11": [1, 9, 18, 19], "12": [3, 4, 8, 12],  "13": [19, 3, 4], "14": [2, 5, 15], "15": [3, 4, 8, 18]}
# dirs = [i for i in range(1, 16)]
# dirs.remove(2)
# dirs.remove(7)
# for d in dirs:
#     single_dir_test_ims = test_ims[str(d)]
#     dir_ims = ["{}_{}".format(d, im) for im in single_dir_test_ims]
#     print(dir_ims)
#     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(d))
#     js_data = json.load(open(js_file, 'r'))
#     for k, v in js_data.items():
#         if k in dir_ims:
#             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
#         else:
#             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
#     plt.show()


# for i in range(1, 22):
#     path = os.path.join(dir, 'data', str(i))
#     ims = os.listdir(path)
#     assert len(ims) == 20

# LAB = json.load(open(r'./1118_lab.json', 'r'))
# RGB = json.load(open(r'./1118_train_rgb.json', 'r'))
#
# L, A, B = [], [], []
# v_k = dict()
# lab_kgb = dict()
# lab_kgr = dict()
#
# rgb_lab = []
# ks = []
# for k, v in LAB.items():
#     k_gb, k_gr, k_rb = RGB[k][1] / RGB[k][2], RGB[k][1] / RGB[k][0], RGB[k][0] / RGB[k][2]
#     rgb_lab.append(RGB[k] + v + [k_gb, k_gr, k_rb])
#     ks.append(k)
#     l,a,b = str(v[0]).split('.')[0], str(v[1]).split('.')[0], str(v[2]).split('.')[0]
#     # if l == '13' and a == '-21' and b == '5':
#     L.append(l)
#     A.append(a)
#     B.append(b)
#     lab_kgb["{}_{}_{}".format(l,a,b)] = gamma(float(RGB[k][1])/255) / gamma(float(RGB[k][2])/255)
#     lab_kgr["{}_{}_{}".format(l, a, b)] = gamma(float(RGB[k][1]) / 255) / gamma(float(RGB[k][0]) / 255)
#
# L = list(set(L))
# A = list(set(A))
# B = list(set(B))
# print("L: {}, A: {}, B: {}".format(L, A, B))
#
#
# # bads = [('10_18', 6), ('9_2', 4), ('7_6', 4), ('9_6', 3), ('9_3', 3), ('8_11', 3), ('8_10', 3), ('2_7', 3), ('14_6', 3), ('9_12', 2), ('8_18', 2), ('8_13', 2), ('7_19', 2), ('7_17', 2), ('7_12', 2), ('7_11', 2), ('7_1', 2), ('6_15', 2), ('3_9', 2), ('3_19', 2), ('3_14', 2), ('2_9', 2), ('1_8', 2), ('1_11', 2), ('14_3', 2), ('14_17', 2), ('14_16', 2), ('14_15', 2), ('14_13', 2), ('13_7', 2), ('13_17', 2), ('13_1', 2), ('10_9', 2), ('10_8', 2), ('9_7', 1), ('9_17', 1), ('9_14', 1), ('9_10', 1), ('8_6', 1), ('8_5', 1), ('8_2', 1), ('8_17', 1), ('8_14', 1), ('8_1', 1), ('7_10', 1), ('6_8', 1), ('6_3', 1), ('6_18', 1), ('6_12', 1), ('5_15', 1), ('5_1', 1), ('3_4', 1), ('3_18', 1), ('3_10', 1), ('2_5', 1), ('2_10', 1), ('1_3', 1), ('1_1', 1), ('14_8', 1), ('14_7', 1), ('14_5', 1), ('14_19', 1), ('14_10', 1), ('13_9', 1), ('13_8', 1), ('13_4', 1), ('13_18', 1), ('13_16', 1), ('13_14', 1), ('10_19', 1), ('10_14', 1)]
# # for bad in bads:
# #     print(LAB[bad[0]])
#
#
# L,A,B = ['12', '13', '11'], ['-21', '-20', '-19', '-18', '-22'], ['4', '1', '3', '2', '7', '5', '0', '6']
# index_lab_str = dict()
# for i in range(3):
#     for j in range(5):
#         for k in range(8):
#             index_lab_str[str(i*40+j*8+k)] = "{}_{}_{}".format(L[i], A[j], B[k])
#
# counts = [0]*120
# for k, v in LAB.items():
#     l,a,b = str(v[0]).split('.')[0], str(v[1]).split('.')[0], str(v[2]).split('.')[0]
#     for i, l_ in enumerate(L):
#         for j, a_ in enumerate(A):
#             for k, b_ in enumerate(B):
#                 if l==l_ and a_ == a and b_ == b:
#                     counts[i*40+j*8+k] += 1
#
# lab_counts = dict()
# for ind, count in enumerate(counts):
#     if count != 0:
#         lab_counts[index_lab_str[str(ind)]] = count
# sored = sorted(lab_counts.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
#
# for L in ["11", "12", "13"]:
#     c = 0
#     for ind, val in enumerate(sored):
#             if val[0].split('_')[0] == L:
#                 c += val[1]
#     print("L值为: {}, 出现的次数: {}".format(L, c))
#
# # for ind, val in enumerate(sored):
# #     print("lab: {}, count: {}, k_gb: {}, k_gr: {}".format(val[0], val[1], lab_kgb[val[0]], lab_kgr[val[0]]))
#
#
#
# from sklearn.cluster import KMeans
# from sklearn import metrics
# clusterer = KMeans(n_clusters=3, random_state=66).fit(rgb_lab)
# centers = clusterer.cluster_centers_
# print(centers)
#
# preds = clusterer.predict(rgb_lab)
# ks1, ks2, ks3 = [], [], []
# ff2 = open(r'./k2.txt', 'w')
# for ind, cls in enumerate(preds):
#     if cls == 0:
#         ks1.append(ks[ind])
#     elif cls == 1:
#         ks2.append(ks[ind])
#         ff2.write(ks[ind]+',')
#     elif cls == 2:
#         ks3.append(ks[ind])
# print(len(ks1), len(ks2), len(ks3))
# print("ks1: {}".format(ks1))
# print('')
# print("ks1: {}".format(ks1))


# centers = clusterer.cluster_centers_
# # 得到簇中心位置
#
# sample_preds = clusterer.predict(pca_samples)
# # 采样数据的预测结果
#
# score = metrics.silhouette_score(reduced_data, preds, metric='euclidean')# 计算轮廓系数的均值
# print(score)

# a = json.load(open(r'/Users/chenjia/Downloads/Learning/SmartMore/1110_beijing/zeiss_rgb2lab-dev/test_data_lab.json', 'r'))
# for k, v in a.items():
#     print(k, v)

# import xlrd
# test_gt = dict()
# test_gt_csv = r'./test_data_gt.xlsx'
# wb = xlrd.open_workbook(test_gt_csv)
# data = wb.sheet_by_name(r'Sheet1')
# rows = data.nrows
# for i in range(1, rows):
#     im_name = data.cell(i, 0).value
#     l,a,b = data.cell(i, 1).value, data.cell(i, 2).value, data.cell(i, 3).value
#     l_pre, a_pre, b_pre = data.cell(i, 5).value, data.cell(i, 6).value, data.cell(i, 7).value
#     # if abs(l_pre-l) >= 0.5 or abs(a_pre-a) >= 0.5 or abs(b_pre-b) >= 0.5:
#     #     if im_name.split('_')[0] not in ['11', '12']:
#     #         print(im_name)
#     test_gt[im_name] = [l,a,b]


# old_pred = json.load(open('./0.json', 'r'))
# for k, v in test_gt.items():
#     pred = [float(a) for a in old_pred[k]]
#     diff = [abs(v[i]-pred[i]) for i in range(3)]
#     for di in diff:
#         if di >= 0.5:
#             print("img: {}, diff: {}".format(k, diff))


def pass_unpass_show_each_oven_rgb():
    def gamma(a):
        if a > 0.04045:
            a = np.power((a + 0.055) / 1.055, 2.4)
        else:
            a /= 12.92

        return a

    inds = [i for i in range(17, 22)]
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'yellow', 'red', 'darkslateblue', 'turquoise',
              'blue', 'yellow', 'black', 'pink', 'red', 'green', 'cornflowerblue', 'purple', 'turquoise']
    # colors = ['pink'] * 15 + ['black', 'red']
    test_rgb = json.load(open(r'./1118_blue_test_rgb.json', 'r'))
    for ind in inds:
        js = json.load(open(r'./1122_rgb_js/dir_{}_rgb.json'.format(ind), 'r'))
        ks = list(js.keys())
        for ii, k in enumerate(ks):
            if ii == 0:
                plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind-10], label=ind)
            else:
                plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind-10])
            if k in test_rgb:
                plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color='red')

    plt.grid()
    plt.legend()
    plt.show()

# pass_unpass_show_each_oven_rgb()


def write_to_csv():
    js_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
    ks, Rs, Gs, Bs = [], [], [], []
    for i in range(1, 16):
        js_path = os.path.join(js_dir, "dir_{}_rgb.json".format(i))
        # print(js_path)
        rgb_data = json.load(open(js_path, 'r'))
        for k, v in rgb_data.items():
            ks.append(k)
            Rs.append(v[0])
            Gs.append(v[1])
            Bs.append(v[2])
    df = pd.DataFrame()
    df['image_name'] = ks
    df['R'] = Rs
    df['G'] = Gs
    df['B'] = Bs
    df.to_csv(r'./1209_green_rgb.csv')

write_to_csv()



