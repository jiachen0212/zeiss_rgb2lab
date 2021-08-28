# coding=utf-8
'''
处理0812客户给到的第二批数据 包含蓝绿色膜
rgb出的6个值  [重叠区域和外围区域? ]
返回rgb6值-lab曲线
rgb6值-lab三个value值

'''
import xlrd
import json

lab_file = r'D:\work\project\卡尔蔡司AR镀膜\poc\0812\0812\自动膜色识别数据记录 2021-08-12.xlsx'
wb = xlrd.open_workbook(lab_file)
data_path = r'D:\work\project\卡尔蔡司AR镀膜\poc\0812\0812'
all_ = dict()
all_lab_value = dict()

for i in range(1, 9):
    data = wb.sheet_by_name('Sheet{}'.format(i))
    rows = data.nrows
    title = data.row_values(2)
    index380, index780 = title.index(380), title.index(780)
    for j in range(4, rows):
        row_data = data.row_values(j)
        lab = row_data[index380: index780+1]
        l, a, b = row_data[3], row_data[4], row_data[5]
        print(l, a, b)
        all_["{}_{}".format(i+20, j-3)] = lab
        all_lab_value["{}_{}".format(i, j-3)] = [l,a,b]

data = json.dumps(all_)
with open('./all_lab_0817.json', 'w') as js_file:
    js_file.write(data)


data = json.dumps(all_lab_value)
with open(r'D:\work\project\卡尔蔡司AR镀膜\poc\0812lab_value.json', 'w') as js_file:
    js_file.write(data)


