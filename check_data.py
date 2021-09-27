# coding=utf-8
import json

lab = json.load(open('./lab_value_0924.json', 'r'))
rgb = json.load(open('./3float_rgb_0924.json', 'r'))   # len == 315

# 其他材质数据
other_lab = dict()
other_rgb = dict()

labs = []
rgbs = []
ks = []
for k, v in lab.items():
    # print("LAB: {}, dir: {}, rgb: {}".format(v, k, rgb[k]))
    labs.append(v)
    rgbs.append(rgb[k])
    ks.append(str(int(k.split('_')[0])-50) + '_' + k.split('_')[1])
    if k.split('_')[0] in ['65', '67', '68']:
        other_rgb[k] = rgb[k]
        other_lab[k] = v
print("other data size: {}".format(len(other_lab)))
data = json.dumps(other_lab)
with open('./other_lab.json', 'w') as js_file:
    js_file.write(data)
data = json.dumps(other_rgb)
with open('./other_rgb.json', 'w') as js_file:
    js_file.write(data)


# 对每一个lab 找与之最相近的lab_1, 然后看看他们之间rgb的差值
f = open(r'./rgb_lab_check.txt', 'w')
diff_labs = []
diff_rgbs = []
xs_dir = []
nums = len(labs)

import numpy as np
for i in range(nums):
    cur_lab = labs[i]
    tt = "dir_name: {}".format(ks[i])
    # print(tt)
    f.write(tt+'\n')
    tmp_lab = labs[:i] + labs[i+1:]
    tmp_rgb = rgbs[:i] + rgbs[i+1:]
    tmp_ks = ks[:i] + ks[i+1:]
    diff_lab = [(abs(tmp_lab[j][0] - cur_lab[0]))+(abs(tmp_lab[j][1] - cur_lab[1]))+(abs(tmp_lab[j][2] - cur_lab[2])) for j in range(nums-1)]
    min_diff_index = diff_lab.index(min(diff_lab))
    a = "cur_lab: {}".format(cur_lab)
    # print(a)
    b = "最相近的lab值: {}".format(tmp_lab[min_diff_index])
    # print(b)
    ff = "最相近的dir_name: {}\n".format(tmp_ks[min_diff_index])
    # print(ff)
    f.write(ff)
    diff_1 = min(diff_lab)
    diff_labs.append(diff_1)
    xs_dir.append(tmp_ks[min_diff_index])
    cc = "diff_lab: {}".format(diff_1)
    # print(cc)
    # print('====')
    cur_rgb = rgbs[i]
    c = "cur_rgb: {}".format(cur_rgb)
    # print(c)
    d = "最相近lab对应的rgb值: {}".format(tmp_rgb[min_diff_index])
    diff_rgb = np.sum([abs(cur_rgb[j]-tmp_rgb[min_diff_index][j]) for j in range(3)])
    diff_rgbs.append(diff_rgb)
    # print(d)
    ss = "diff_rgb: {}".format(diff_rgb)
    # print(ss)
    # print('\n')
    f.write(a+'\n')
    f.write(b + '\n')
    f.write("======" + '\n')
    f.write(c + '\n')
    f.write(d + '\n')
    f.write(cc + '\n')
    f.write(ss + '\n')
    f.write('\n')

# lab的diff在0.5以内, rgb的diff却>15.. 是不是有点异常了啊..?
blue_color = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_color.json', 'r'))
blue_lab = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_lab.json', 'r'))
# blue_color = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\green_0924_backup\green_color.json', 'r'))
# blue_lab = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\green_0924_backup\green_lab.json', 'r'))
print(len(blue_color))
c = 0
assert len(diff_labs) == len(diff_rgbs)
bad = ['5_8', '6_4', '8_5', '8_8', '10_16', '11_5', '12_2', '15_5', '18_2', '18_4', '18_6', '18_15', '18_16',
       '18_17', '19_16', '19_17']
for i in range(len(diff_labs)):
    # if diff_rgbs[i] > 10 and diff_labs[i] < 0.5:
    #     c += 1
    #     a = "dir_name: {}, 相近的dir_name: {}, diff_lab: {}, diff_rgb: {}".format(ks[i], xs_dir[i], diff_labs[i], np.sum(diff_rgbs[i]))
    #     print(a)
    #     f.write(a+'\n')
    #     ttmp = str(int(ks[i].split('_')[0])+50) + '_' + ks[i].split('_')[1]
    #     try:
    #         del blue_color[ttmp]
    #         del blue_lab[ttmp]
    #     except:
    #         continue

    if ks[i] in bad:
        ttmp = str(int(ks[i].split('_')[0]) + 50) + '_' + ks[i].split('_')[1]
        try:
            del blue_color[ttmp]
            del blue_lab[ttmp]
            c += 1
        except:
            continue


print("lab diff in 0.5, rgb diff > 10: {}".format(c))
print(len(blue_color))
data = json.dumps(blue_color)
with open('./blue_color.json', 'w') as js_file:
    js_file.write(data)

data = json.dumps(blue_lab)
with open('./blue_lab.json', 'w') as js_file:
    js_file.write(data)

