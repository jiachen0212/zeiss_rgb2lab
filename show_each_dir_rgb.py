# # # coding=utf-8
# # '''
# # 统计lab值分布情况 和 对应的g/b
# #
# # '''
# #
# #
# #
# # # import json
# # # import pandas as pd
# # #
# # # a = json.load(open('./1118_test_rgb.json', 'r'))
# # # df = pd.DataFrame()
# # # ks, vs = [], []
# # # for k, v in a.items():
# # #     ks.append(k)
# # #     vs.append(v[:-1])
# # # df['img_name'] = ks
# # # df['RGB'] = vs
# # #
# # # df.to_csv(r'./test_data_rgb.csv')

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

colors = ['aliceblue', 'antiquewhite', 'aliceblue', 'pink', 'red', 'darkslateblue', 'pink',
              'blue', 'yellow', 'black', 'turquoise', 'green', 'cornflowerblue', 'purple', 'pink', 'black', 'cornflowerblue', 'yellow', 'green', 'cornflowerblue', 'purple', 'turquoise']
# colors = ['pink']*16+ ['cornflowerblue', 'yellow', 'green']
# dir = r'D:\work\project\卡尔蔡司膜色缺陷\0107'
# rgb_jss = os.listdir(dir)
# js = [a for a in rgb_jss if '.json' in a]
# plt.title("green")
# for i in range(1, 19):
#     js_file = r'dir_{}_rgb.json'.format(i)
#     js_data = json.load(open(os.path.join(dir, js_file), 'r'))
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

# data0107 = dict()
# for i in range(1, 19):
#     js_file = r'dir_{}_rgb.json'.format(i)
#     js_data = json.load(open(os.path.join(dir, js_file), 'r'))
#     for k, v in js_data.items():
#         # k_ = "{}_{}".format(int(k.split('_')[0])-11, k.split('_')[1])
#         data0107[k] = v
# print(len(data0107))
# data = json.dumps(data0107)
# with open(r'./0107/0107rgb.json', 'w') as js_file:
#     js_file.write(data)

# for i in range(1, 19):
#     for k, v in data0107.items():
#         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
# plt.grid()
# plt.show()

# df = pd.DataFrame()
# ims, Rs, Gs, Bs = [], [], [], []
# for k, v in data0107.items():
#     ims.append(k)
#     Rs.append(v[0])
#     Gs.append(v[1])
#     Bs.append(v[2])
# df["img"] = ims
# df["R"] = Rs
# df["G"] = Gs
# df["B"] = Bs
# df.to_csv('./0107RGB.csv')

seg_rgb = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\0107\0107all.json', 'r'))
# ind = 0
# for i in range(1, 19):
#     for k, v in seg_rgb.items():
#         if k.split('_')[0] == str(i):
#             if ind == 0:
#                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
#                 ind += 1
#             else:
#                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
#     plt.legend()
#     plt.grid()
#     plt.show()


inds = [0]*18
rgb3 = []
for k, v in seg_rgb.items():
    if (k.split('_')[0] == "3") or (k.split('_')[0] == "10"):
        if inds[0] == 0:
            plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color="blue", label="dir_3/10")
            rgb3.append(v)
            inds[0] += 1
        else:
            rgb3.append(v)
            plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color="blue")
    else:
        plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color="pink")
plt.grid()
plt.legend()
plt.show()
# for v in rgb3:
#     if float(v[1]) < float(v[2]):
#         print(v)
    # elif k.split('_')[0] == "2":
    #     if inds[1] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[1] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "3":
    #     if inds[2] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[2] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "4":
    #     if inds[3] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[3] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "5":
    #     if inds[4] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[4] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "6":
    #     if inds[5] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[5] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "7":
    #     if inds[6] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[6] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "8":
    #     if inds[7] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[7] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # if k.split('_')[0] == "9":
    #     if inds[8] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[8] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])

    # elif k.split('_')[0] == "10":
    #     if inds[9] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[9] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "11":
    #     if inds[10] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[10] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "12":
    #     if inds[11] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[11] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "13":
    #     if inds[12] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[12] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "14":
    #     if inds[13] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[13] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "15":
    #     if inds[14] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[14] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "16":
    #     if inds[15] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[15] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "17":
    #     if inds[16] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[16] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
    #
    # elif k.split('_')[0] == "18":
    #     if inds[17] == 0:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])], label="dir_{}".format(k.split('_')[0]))
    #         inds[17] += 1
    #     else:
    #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])




#     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[int(k.split('_')[0])])
# plt.legend()
# plt.grid()
# plt.show()
#
#
# import random
# tests = []
# nums = [7]*15
# nums[11], nums[12] = 6, 6
# for i in range(1, 16):
#     a = [i for i in range(15)]
#     random.shuffle(a)
#     tests.extend(["{}_{}".format(i, b) for b in a[:7]])
#
# test = ['1_4', '1_6', '1_7', '1_11', '1_13', '1_0', '2_8', '2_14', '2_13', '2_10', '2_3', '2_9', '2_7', '3_9', '3_0', '3_12', '3_11', '3_10', '3_1', '4_13', '4_0', '4_12', '4_2', '4_6', '4_4', '4_14', '5_12', '5_10', '5_8', '5_1', '5_13', '5_0', '5_20', '6_6', '6_5', '6_2', '6_13', '6_4', '6_14', '6_12', '7_13', '7_12', '7_3', '7_0', '7_9', '7_7', '7_5', '8_1', '8_0', '8_10', '8_20', '8_5', '8_2', '8_3', '8_14', '9_11', '9_3', '9_1', '9_13', '9_6', '9_9', '9_5', '10_1', '10_10', '10_0', '10_14', '10_12', '10_9', '10_2', '11_3', '11_2', '11_8', '11_0', '11_13', '11_9', '11_14', '12_2', '12_8', '12_6', '12_7', '12_15', '12_20', '13_4', '13_12', '13_5', '13_14', '14_6', '14_14', '14_8', '14_0', '14_10', '14_4', '14_2', '15_11', '15_2', '15_9', '15_8', '15_0', '15_3', '15_7', '16_3', '17_3', '18_3']
# print(len(test))
#
# for k, v in seg_rgb.items():
#     if k not in test:
#         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
#     else:
#         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# plt.grid()
# plt.show()
#
#
#
# for k, v in seg_rgb.items():
#     if k.split("_")[0] != "3":
#         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
#     else:
#         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# plt.grid()
# plt.show()





# # base_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
# # rgb_jss = os.listdir(base_dir)
# # js = [a for a in rgb_jss if '.json' in a]
# #
# # # for i3 in range(1, 16):
# # #     for i in range(1, 16):
# # #         js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
# # #         js_data = json.load(open(js_file, 'r'))
# # #         assert len(js_data) == 20
# # #         for k, v in js_data.items():
# # #             if i == i3:
# # #                 plt.plot([0, 1, 2],[gamma(float(a) / 255) for a in v], color='red')
# # #             else:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
# # #     plt.title("dir{}_and_others".format(i3))
# # #     plt.legend()
# # #     plt.grid()
# # #     plt.savefig('./文件夹颜色差异/{}.png'.format(i3))
# # #     # plt.show()
# # #     plt.close()
# #
# # dirs = [i for i in range(1, 16)]
# # green_test = json.load(open(r'./1209_test_rgb.json', 'r'))
# # for i in dirs:
# #     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
# #     js_data = json.load(open(js_file, 'r'))
# #     assert len(js_data) == 20
# #     index = 0
# #     for k, v in js_data.items():
# #         if k in green_test:
# #             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# #         elif k in ['13_18', '11_4', '2_11', '13_7', '13_20']:
# #             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='blue')
# #         # else:
# #         #     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
# #         index += 1
# # plt.legend()
# # plt.grid()
# # # plt.savefig('./文件夹颜色差异/each_dir/{}.png'.format(i))
# # plt.show()
# # # plt.close()
# #
# #
# # # for i in range(16,22):
# # #     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(i))
# # #     js_data = json.load(open(js_file, 'r'))
# # #     assert len(js_data) == 20
# # #     keys = list(js_data.keys())
# # #     for test_index in range(20):
# # #         if i not in [2, 7]:
# # #             dir_ = os.path.join(r'D:\work\project\卡尔蔡司膜色缺陷\文件夹颜色差异\each_dir', str(i))
# # #             if not os.path.exists(dir_):
# # #                 os.makedirs(dir_)
# # #             for index, key in enumerate(keys):
# # #                 if index == test_index:
# # #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js_data[key]], color='red',
# # #                              label="test_im: {}".format(key))
# # #                 else:
# # #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js_data[key]], color='cornflowerblue')
# # #             plt.legend()
# # #             plt.grid()
# # #             plt.show()
# # #             # plt.savefig(os.path.join(dir_, '{}.png'.format(test_index)))
# # #             # plt.close()
# #
# #
# # # green
# # # test_ims = {"1": [19,20,2,4], "3": [2,3,10,14], "4": [8,9,19,20], "5": [2,20,5,11], "6": [4,5,18,20], "8": [1,4, 9, 15], "9": [1, 12, 10, 14], "10": [1, 2, 3, 12] , "11": [1, 9, 18, 19], "12": [3, 4, 8, 12],  "13": [19, 3, 4], "14": [2, 5, 15], "15": [3, 4, 8, 18]}
# # # dirs = [i for i in range(1, 16)]
# # # dirs.remove(2)
# # # dirs.remove(7)
# # # for d in dirs:
# # #     single_dir_test_ims = test_ims[str(d)]
# # #     dir_ims = ["{}_{}".format(d, im) for im in single_dir_test_ims]
# # #     print(dir_ims)
# # #     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(d))
# # #     js_data = json.load(open(js_file, 'r'))
# # #     for k, v in js_data.items():
# # #         if k in dir_ims:
# # #             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# # #         else:
# # #             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
# # #     plt.show()
# #
# #
# # # for i in range(1, 22):
# # #     path = os.path.join(dir, 'data', str(i))
# # #     ims = os.listdir(path)
# # #     assert len(ims) == 20
# #
# # # LAB = json.load(open(r'./1118_lab.json', 'r'))
# # # RGB = json.load(open(r'./1118_train_rgb.json', 'r'))
# # #
# # # L, A, B = [], [], []
# # # v_k = dict()
# # # lab_kgb = dict()
# # # lab_kgr = dict()
# # #
# # # rgb_lab = []
# # # ks = []
# # # for k, v in LAB.items():
# # #     k_gb, k_gr, k_rb = RGB[k][1] / RGB[k][2], RGB[k][1] / RGB[k][0], RGB[k][0] / RGB[k][2]
# # #     rgb_lab.append(RGB[k] + v + [k_gb, k_gr, k_rb])
# # #     ks.append(k)
# # #     l,a,b = str(v[0]).split('.')[0], str(v[1]).split('.')[0], str(v[2]).split('.')[0]
# # #     # if l == '13' and a == '-21' and b == '5':
# # #     L.append(l)
# # #     A.append(a)
# # #     B.append(b)
# # #     lab_kgb["{}_{}_{}".format(l,a,b)] = gamma(float(RGB[k][1])/255) / gamma(float(RGB[k][2])/255)
# # #     lab_kgr["{}_{}_{}".format(l, a, b)] = gamma(float(RGB[k][1]) / 255) / gamma(float(RGB[k][0]) / 255)
# # #
# # # L = list(set(L))
# # # A = list(set(A))
# # # B = list(set(B))
# # # print("L: {}, A: {}, B: {}".format(L, A, B))
# # #
# # #
# # # # bads = [('10_18', 6), ('9_2', 4), ('7_6', 4), ('9_6', 3), ('9_3', 3), ('8_11', 3), ('8_10', 3), ('2_7', 3), ('14_6', 3), ('9_12', 2), ('8_18', 2), ('8_13', 2), ('7_19', 2), ('7_17', 2), ('7_12', 2), ('7_11', 2), ('7_1', 2), ('6_15', 2), ('3_9', 2), ('3_19', 2), ('3_14', 2), ('2_9', 2), ('1_8', 2), ('1_11', 2), ('14_3', 2), ('14_17', 2), ('14_16', 2), ('14_15', 2), ('14_13', 2), ('13_7', 2), ('13_17', 2), ('13_1', 2), ('10_9', 2), ('10_8', 2), ('9_7', 1), ('9_17', 1), ('9_14', 1), ('9_10', 1), ('8_6', 1), ('8_5', 1), ('8_2', 1), ('8_17', 1), ('8_14', 1), ('8_1', 1), ('7_10', 1), ('6_8', 1), ('6_3', 1), ('6_18', 1), ('6_12', 1), ('5_15', 1), ('5_1', 1), ('3_4', 1), ('3_18', 1), ('3_10', 1), ('2_5', 1), ('2_10', 1), ('1_3', 1), ('1_1', 1), ('14_8', 1), ('14_7', 1), ('14_5', 1), ('14_19', 1), ('14_10', 1), ('13_9', 1), ('13_8', 1), ('13_4', 1), ('13_18', 1), ('13_16', 1), ('13_14', 1), ('10_19', 1), ('10_14', 1)]
# # # # for bad in bads:
# # # #     print(LAB[bad[0]])
# # #
# # #
# # # L,A,B = ['12', '13', '11'], ['-21', '-20', '-19', '-18', '-22'], ['4', '1', '3', '2', '7', '5', '0', '6']
# # # index_lab_str = dict()
# # # for i in range(3):
# # #     for j in range(5):
# # #         for k in range(8):
# # #             index_lab_str[str(i*40+j*8+k)] = "{}_{}_{}".format(L[i], A[j], B[k])
# # #
# # # counts = [0]*120
# # # for k, v in LAB.items():
# # #     l,a,b = str(v[0]).split('.')[0], str(v[1]).split('.')[0], str(v[2]).split('.')[0]
# # #     for i, l_ in enumerate(L):
# # #         for j, a_ in enumerate(A):
# # #             for k, b_ in enumerate(B):
# # #                 if l==l_ and a_ == a and b_ == b:
# # #                     counts[i*40+j*8+k] += 1
# # #
# # # lab_counts = dict()
# # # for ind, count in enumerate(counts):
# # #     if count != 0:
# # #         lab_counts[index_lab_str[str(ind)]] = count
# # # sored = sorted(lab_counts.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
# # #
# # # for L in ["11", "12", "13"]:
# # #     c = 0
# # #     for ind, val in enumerate(sored):
# # #             if val[0].split('_')[0] == L:
# # #                 c += val[1]
# # #     print("L值为: {}, 出现的次数: {}".format(L, c))
# # #
# # # # for ind, val in enumerate(sored):
# # # #     print("lab: {}, count: {}, k_gb: {}, k_gr: {}".format(val[0], val[1], lab_kgb[val[0]], lab_kgr[val[0]]))
# # #
# # #
# # #
# # # from sklearn.cluster import KMeans
# # # from sklearn import metrics
# # # clusterer = KMeans(n_clusters=3, random_state=66).fit(rgb_lab)
# # # centers = clusterer.cluster_centers_
# # # print(centers)
# # #
# # # preds = clusterer.predict(rgb_lab)
# # # ks1, ks2, ks3 = [], [], []
# # # ff2 = open(r'./k2.txt', 'w')
# # # for ind, cls in enumerate(preds):
# # #     if cls == 0:
# # #         ks1.append(ks[ind])
# # #     elif cls == 1:
# # #         ks2.append(ks[ind])
# # #         ff2.write(ks[ind]+',')
# # #     elif cls == 2:
# # #         ks3.append(ks[ind])
# # # print(len(ks1), len(ks2), len(ks3))
# # # print("ks1: {}".format(ks1))
# # # print('')
# # # print("ks1: {}".format(ks1))
# #
# #
# # # centers = clusterer.cluster_centers_
# # # # 得到簇中心位置
# # #
# # # sample_preds = clusterer.predict(pca_samples)
# # # # 采样数据的预测结果
# # #
# # # score = metrics.silhouette_score(reduced_data, preds, metric='euclidean')# 计算轮廓系数的均值
# # # print(score)
# #
# # # a = json.load(open(r'/Users/chenjia/Downloads/Learning/SmartMore/1110_beijing/zeiss_rgb2lab-dev/test_data_lab.json', 'r'))
# # # for k, v in a.items():
# # #     print(k, v)
# #
# # # import xlrd
# # # test_gt = dict()
# # # test_gt_csv = r'./test_data_gt.xlsx'
# # # wb = xlrd.open_workbook(test_gt_csv)
# # # data = wb.sheet_by_name(r'Sheet1')
# # # rows = data.nrows
# # # for i in range(1, rows):
# # #     im_name = data.cell(i, 0).value
# # #     l,a,b = data.cell(i, 1).value, data.cell(i, 2).value, data.cell(i, 3).value
# # #     l_pre, a_pre, b_pre = data.cell(i, 5).value, data.cell(i, 6).value, data.cell(i, 7).value
# # #     # if abs(l_pre-l) >= 0.5 or abs(a_pre-a) >= 0.5 or abs(b_pre-b) >= 0.5:
# # #     #     if im_name.split('_')[0] not in ['11', '12']:
# # #     #         print(im_name)
# # #     test_gt[im_name] = [l,a,b]
# #
# #
# # # old_pred = json.load(open('./0.json', 'r'))
# # # for k, v in test_gt.items():
# # #     pred = [float(a) for a in old_pred[k]]
# # #     diff = [abs(v[i]-pred[i]) for i in range(3)]
# # #     for di in diff:
# # #         if di >= 0.5:
# # #             print("img: {}, diff: {}".format(k, diff))
# #
# #
# # def pass_unpass_show_each_oven_rgb():
# #     def gamma(a):
# #         if a > 0.04045:
# #             a = np.power((a + 0.055) / 1.055, 2.4)
# #         else:
# #             a /= 12.92
# #
# #         return a
# #
# #     inds = [i for i in range(17, 22)]
# #     colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'yellow', 'red', 'darkslateblue', 'turquoise',
# #               'blue', 'yellow', 'black', 'pink', 'red', 'green', 'cornflowerblue', 'purple', 'turquoise']
# #     # colors = ['pink'] * 15 + ['black', 'red']
# #     test_rgb = json.load(open(r'./1118_blue_test_rgb.json', 'r'))
# #     for ind in inds:
# #         js = json.load(open(r'./1122_rgb_js/dir_{}_rgb.json'.format(ind), 'r'))
# #         ks = list(js.keys())
# #         for ii, k in enumerate(ks):
# #             if ii == 0:
# #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind-10], label=ind)
# #             else:
# #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind-10])
# #             if k in test_rgb:
# #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color='red')
# #
# #     plt.grid()
# #     plt.legend()
# #     # plt.show()
# #
# # # pass_unpass_show_each_oven_rgb()
# #
# #
# # def write_to_csv():
# #     js_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
# #     ks, Rs, Gs, Bs = [], [], [], []
# #     for i in range(1, 16):
# #         js_path = os.path.join(js_dir, "dir_{}_rgb.json".format(i))
# #         # print(js_path)
# #         rgb_data = json.load(open(js_path, 'r'))
# #         for k, v in rgb_data.items():
# #             ks.append(k)
# #             Rs.append(v[0])
# #             Gs.append(v[1])
# #             Bs.append(v[2])
# #     df = pd.DataFrame()
# #     df['image_name'] = ks
# #     df['R'] = Rs
# #     df['G'] = Gs
# #     df['B'] = Bs
# #     df.to_csv(r'./1209_green_rgb.csv')
# #
# # # write_to_csv()
# #
# #
# # import xlrd
# # ff = open('./res.txt', 'w')
# # L = [9.5, 14.5]
# # A = [-24, -15]
# # B = [-2, 10]
# # test_data_gt = dict()
# # wb = xlrd.open_workbook(r'./1209_green_data_lab_reslut2.xlsx')
# # data = wb.sheet_by_name(r'Sheet1')
# # rows = data.nrows
# # c,d = 0, 0
# # for i in range(1, rows):
# #     pass_or_unpass = ''
# #     line = data.row_values(i)
# #     real_l, real_a, real_b = line[6], line[7], line[8]
# #     test_data_gt[line[1]] = [real_l, real_a, real_b]
# #     if (L[0] <= real_l <= L[1]) and (A[0] <= real_a<= A[1]) and (B[0] <= real_b <= B[1]):
# #         pass_or_unpass = True
# #     else:
# #         pass_or_unpass = False
# #     diffl, diffa, diffb = line[10], line[11], line[12]
# #     if abs(diffl) >= 0.5 or abs(diffa) >= 0.5 or abs(diffb) >= 0.5:
# #         c += 1
# #         info = "img_name: {}, diff: {}, LAB_range: {}".format(line[1], [diffl, diffa, diffb], pass_or_unpass)
# #         ff.write(info+'\n')
# #         if not pass_or_unpass:
# #             d += 1
# #             print(line[1], 'LAB异常')
# #
# # print(c, d)
# # # print(test_data_gt)
# # data = json.dumps(test_data_gt)
# # with open(r'./1209_green_test_lab.json', 'w') as js_file:
# #     js_file.write(data)
# #
# #
# # # c, d = 0, 0
# # # seed_mean_pred_test_lab = json.load(open(r'./mean_pred_test_lab.json', 'r'))
# # # for k, v in test_data_gt.items():
# # #     pass_ = False
# # #     if (L[0] <= v[0] <= L[1]) and (A[0] <= v[1] <= A[1]) and (B[0] <= v[2] <= B[1]):
# # #         pass_ = True
# # #     diff = [v[i]-seed_mean_pred_test_lab[k][i] for i in range(3)]
# # #     if abs(diff[0]) >= 0.5 or abs(diff[1]) >= 0.5 or abs(diff[2]) >= 0.5:
# # #         c += 1
# # #         if not pass_:
# # #             d += 1
# # #             print(k, 'LAB异常')
# # # print(c, d)
# #
# #
# # bad1 = ['4_2', '4_6', '9_7', '14_6', '14_16', '14_8']
# # bad2 = ['3_11', '4_1','4_12','4_15','4_16','8_6','10_15','11_14','12_7','13_16','14_2','14_11','14_13','14_14','14_15','14_17']
# # bad3 = ['15_9', '15_10', '5_18', '13_15', '9_18']
# # all_bad = bad3 + bad2 + bad1
# #
# # inds = [i for i in range(1, 16)]
# # colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'yellow', 'red', 'darkslateblue', 'turquoise',
# #           'blue', 'yellow', 'black', 'pink', 'red', 'green', 'cornflowerblue', 'purple', 'turquoise']
# # ind1, ind2, ind3, ind4 = 0,0,0, 0
# # index = 0
# # train_rgb = json.load(open(r'./1209_train_rgb.json', 'r'))
# # test_rgb = json.load(open(r'./1209_test_rgb.json', 'r'))
# # # for ind in inds:
# # #     js = json.load(open(r'./1209/dir_{}_rgb.json'.format(ind), 'r'))
# # #     for k, v in js.items():
# # #         if k in train_rgb:
# # #             if ind4 == 0:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink', label='train data')
# # #                 ind4 += 1
# # #             else:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
# # #         elif k in bad1:
# # #             if ind1 == 0:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='blue', label='diff<=0.6')
# # #                 ind1 += 1
# # #             else:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='blue')
# # #         elif k in bad2:
# # #             if ind2 == 0:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='yellow', label='diff>0.6')
# # #                 ind2 += 1
# # #             else:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='yellow')
# # #         elif k in bad3:
# # #             if ind3 == 0:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red', label='bad LAB_Range')
# # #                 ind3 += 1
# # #             else:
# # #                 plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# # #         # else:
# # #         #     if index == 0:
# # #         #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue', label='ok_test')
# # #         #         index += 1
# # #         #     else:
# # #         #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
# # # plt.legend()
# # # plt.grid()
# # # plt.show()
# #
# # a,b,c = 0,0,0
# # bad = ['3_11','8_6','10_15','11_14','12_7', '13_16']
# # for ind in inds:
# #     js = json.load(open(r'./1209/dir_{}_rgb.json'.format(ind), 'r'))
# #     for k, v in js.items():
# #         if k in train_rgb:
# #         #     if a == 0:
# #         #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink', label='train data')
# #         #         a += 1
# #         #     else:
# #         #         plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='pink')
# #             pass
# #         elif k in test_rgb:
# #             if k not in bad:
# #                 if b == 0:
# #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue', label='test data')
# #                     b += 1
# #                 else:
# #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='cornflowerblue')
# #
# #             else:
# #                 if c == 0:
# #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red', label='not 4 14, diff>0.6')
# #                     c += 1
# #                 else:
# #                     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')
# # plt.legend()
# # plt.grid()
# # plt.show()
# #
# #
# #
# # # rgb lab的值, 一起对照输出一份?
# # train_lab = json.load(open(r'./1209_green_lab.json', 'r'))
# # train_rgb = json.load(open(r'./1209_train_rgb.json', 'r'))
# # test_rgb = json.load(open(r'./1209_test_rgb.json', 'r'))
# # # test_data_gt
# # ks, labs, rgbs = [], [], []
# # for k, v in train_lab.items():
# #     ks.append(k)
# #     labs.append(v)
# #     rgbs.append(train_rgb[k])
# # for k, v in test_data_gt.items():
# #     ks.append(k)
# #     labs.append(v)
# #     rgbs.append(test_rgb[k])
# #
# # Rs = [a[0] for a in rgbs]
# # Gs = [a[1] for a in rgbs]
# # Bs = [a[2] for a in rgbs]
# # L = [a[0] for a in labs]
# # A = [a[1] for a in labs]
# # B = [a[2] for a in labs]
# # df = pd.DataFrame()
# # df["im_name"] = ks
# # df["R"] = Rs
# # df["G"] = Gs
# # df["B"] = Bs
# # df["L_"] = L
# # df["A_"] = A
# # df["B_"] = B
# # df.to_csv(r'./all_data_lab_rgb.csv')
#
#
#
# dirs = [i for i in range(1, 16)]
# import os
# import json
# base_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209'
# for ind, d in enumerate(dirs):
#     js_file = os.path.join(base_dir, 'dir_{}_rgb.json'.format(d))
#     js_data = json.load(open(js_file, 'r'))
#     ii = 0
#     for k, v in js_data.items():
#         if ii == 0:
#             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[ind], label="dir_{}".format(d))
#             ii += 1
#         else:
#             plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[ind])
# plt.legend()
# plt.show()

import xlrd
file = r'D:\work\project\卡尔蔡司膜色缺陷\0107\0107res_diff.xlsx'
test_lab = dict()
b = xlrd.open_workbook(file)
data = b.sheet_by_name("Sheet1")
rows = data.nrows
ind1, ind2, ind3 = data.row_values(0).index('L'), data.row_values(0).index('A'), data.row_values(0).index('B')
i1, i2, i3 = data.row_values(0).index('L*'), data.row_values(0).index('A*'), data.row_values(0).index('B*')
for j in range(1, rows):
    row_data = data.row_values(j)
    if row_data[ind1] != '':
        im_name = row_data[0]
        test_lab[im_name] = [ row_data[i1], row_data[i2], row_data[i3]]
        diffl, diffa, diffb = row_data[ind1], row_data[ind2], row_data[ind3]
        if abs(float(diffl)) >= 0.5 or abs(float(diffa)) >= 0.65 or abs(float(diffb)) >= 0.53:
            print("im_name: {}, diffl: {}, diffa: {}, diffb: {}".format(im_name, diffl, diffa, diffb))

data = json.dumps(test_lab)
with open(r'./0107_test_gt_lab.json', 'w') as js_file:
    js_file.write(data)