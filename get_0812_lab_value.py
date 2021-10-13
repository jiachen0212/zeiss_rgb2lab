# coding=utf-8
'''
处理0812客户给到的第二批数据 包含蓝绿色膜
rgb出的6个值  [重叠区域和外围区域? ]
返回rgb6值-lab曲线
rgb6值-lab三个value值

'''
import xlrd
import json

lab_file = r'D:\work\project\卡尔蔡司膜色缺陷\data\0812\自动膜色识别数据记录 2021-08-12.xlsx'
wb = xlrd.open_workbook(lab_file)
data_path = r'D:\work\project\卡尔蔡司膜色缺陷\data\0812'
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
        # all_["{}_{}".format(i+20, j-3)] = lab
        all_lab_value["{}_{}".format(i+6, j-3)] = [l,a,b]

# data = json.dumps(all_)
# with open('./all_lab_0817.json', 'w') as js_file:
#     js_file.write(data)

data = json.dumps(all_lab_value)
with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812lab.json', 'w') as js_file:
    js_file.write(data)





# # # 0901 组合lab三值json
# data_js1 = json.load(open(r'./0812lab_value.json', 'r'))
# data_js2 = json.load(open(r'./all_lab_value.json', 'r'))
# print(len(data_js1), len(data_js2))
# data_js2_ = dict()
# for k, v in data_js1.items():
#     pre, ind = k.split('_')[0], k.split('_')[1]
#     data_js2_["{}_{}".format(int(pre)+20, ind)] = v
#
# data = json.dumps(data_js2_)
# with open(r'./0812lab_value1.json', 'w') as js_file:
#     js_file.write(data)