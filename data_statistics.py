# coding=utf-8
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gamma(a):
    if a > 0.04045:
        a = np.power((a+0.055)/1.055, 2.4)
    else:
        a /= 12.92

    return a


rgb = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_color.json', 'r'))
lab = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_lab.json', 'r'))
# rgb = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\green_color.json', 'r'))
# lab = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\green_lab.json', 'r'))

blue_data_0924 = pd.DataFrame()

im_names = []
rgbs = []
gamma_ed = []
labs = []
gamma_ed_b_sub_g = []

for k, v in rgb.items():
    im_names.append(str(int(k.split('_')[0])-50) + '_' + k.split('_')[1])
    rgbs.append(v)
    gamma_r, gamma_g, gamma_b = gamma(v[0]/255), gamma(v[1]/255), gamma(v[2]/255)
    gamma_ed_b_sub_g.append(gamma_b - gamma_g)
    gamma_ed.append([gamma_r, gamma_g, gamma_b])
    labs.append(lab[k])

blue_data_0924['dir_name'] = im_names
blue_data_0924['RGB'] = rgbs
blue_data_0924['gamma_ed_RGB'] = gamma_ed
blue_data_0924['LAB'] = labs

# print(blue_data_0924)
blue_data_0924.to_csv('./0924_blue_data.csv', index=False)
# plt.scatter([a for a in range(len(gamma_ed_b_sub_g))], gamma_ed_b_sub_g)
# plt.show()


flag = 1
aa, bb = [], []
plt.title('gammaed_b - gammaed_g')
for ind, gamma in enumerate(gamma_ed):
    if gamma[2] <= 0.6:
        aa.append(gamma_ed_b_sub_g[ind])
    else:
        bb.append(gamma_ed_b_sub_g[ind])
plt.scatter([a for a in range(len(aa))], aa, color='pink', label='gamma_b<=0.6')
plt.scatter([a for a in range(len(bb))], bb, color='blue', label='gamma_b>0.6')
plt.grid()
plt.show()


# # gamma_ed_b > 0.6的, 画一下rgb的三点图 lab的散点图
# plt.subplot(121)
# plt.title('gammaed_b <= 0.6')
# plt.xlabel('RGB')
# aa = [i for i in range(3)]
# flag = 1
# for ind, gamma in enumerate(gamma_ed):
#     if gamma[2] <= 0.6:
#         if flag:
#             plt.scatter(aa, labs[ind], color='blue', label='lab value')
#             # plt.scatter(aa, [a/255 for a in rgbs[ind]], color='pink', label='rgb value')
#         else:
#             plt.scatter(aa, labs[ind], color='blue')
#             # plt.scatter(aa, [a/255 for a in rgbs[ind]], color='pink')
#         flag = 0
# plt.legend()
# plt.grid()
#
# plt.subplot(122)
# plt.title('gammaed_b > 0.6')
# plt.xlabel('RGB')
# aa = [i for i in range(3)]
# flag = 1
# for ind, gamma in enumerate(gamma_ed):
#     if gamma[2] > 0.6:
#         if flag:
#             plt.scatter(aa, labs[ind], color='blue', label='lab value')
#             # plt.scatter(aa, [a/255 for a in rgbs[ind]], color='pink', label='rgb value')
#         else:
#             plt.scatter(aa, labs[ind], color='blue')
#             # plt.scatter(aa, [a/255 for a in rgbs[ind]], color='pink')
#         flag = 0
# plt.legend()
# plt.grid()
#
# plt.show()
