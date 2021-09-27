# coding=utf-8
import json

js_x = json.load(open(r'./all_data_rgb3.json', 'r'))
js_y = json.load(open(r'./all_data_lab.json', 'r'))

js_x_0924 = json.load(open(r'./green_color.json', 'r'))
js_y_0924 = json.load(open(r'./green_lab.json', 'r'))

js_x1 = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_color.json', 'r'))
js_y1 = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_lab.json', 'r'))

pre_green_color = dict()
pre_blue_color = dict()
pre_green_lab = dict()
pre_blue_lab = dict()

for k, v in js_x.items():
    dir_index = int(k.split('_')[0])
    if dir_index <= 21 or k in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
        pre_green_color[k] = v
        pre_green_lab[k] = js_y[k]
    else:
        pre_blue_color[k] = v
        pre_blue_lab[k] = js_y[k]

# all green data
all_green_color = dict(pre_green_color, **js_x_0924)
all_green_lab = dict(pre_green_lab, **js_y_0924)

# all blue data
print("pre_blue_color size: {}".format(len(pre_blue_color)))
all_blue_color = dict(pre_blue_color, **js_x1)
all_blue_lab = dict(pre_blue_lab, **js_y1)
print("all_blue_lab size: {}".format(len(all_blue_lab)))


assert len(all_green_lab) == len(all_green_color)
data = json.dumps(all_green_color)
with open('./all_green_color.json', 'w') as js_file:
    js_file.write(data)

data = json.dumps(all_green_lab)
with open('./all_green_lab.json', 'w') as js_file:
    js_file.write(data)


data = json.dumps(all_blue_color)
with open('./all_blue_color.json', 'w') as js_file:
    js_file.write(data)

data = json.dumps(all_blue_lab)
with open('./all_blue_lab.json', 'w') as js_file:
    js_file.write(data)