# coding=utf-8
'''
处理第一批给到的数据 1-6个文件夹
D:\work\project\卡尔蔡司AR镀膜\poc\膜色图像数据
返回rgb3值-lab3值

'''

import os
import xlrd
import json


base_dir = r'D:\work\project\卡尔蔡司AR镀膜\poc'
lab_pare = r'自动膜色识别数据记录2.xlsx'
# 文件夹1~6数据可用, 文件和图像数据是按顺序对应的
wb = xlrd.open_workbook(os.path.join(base_dir, lab_pare))
data = wb.sheet_by_name('Sheet1')
rows = data.nrows
l_index, b_index = 3, 5
col_data = data.col_values(0)
sub_start = [r'试样1278', '试样1295', '试样1315', '试样1334', '试样1354', '试样1374']
nums = [17, 20, 19, 20, 20, 20]
all_ = dict()
dir_indexs = []
for info in sub_start:
    dir_indexs.append(col_data.index(info))

for ind1, start in enumerate(dir_indexs):
    tmp = range(start, start+nums[ind1])
    for ind2, i in enumerate(tmp):
        all_["{}_{}".format(ind1+1, ind2+1)] = data.row_values(i)[l_index: b_index+1]

data = json.dumps(all_)
with open('./all_lab_value.json', 'w') as js_file:
    js_file.write(data)
