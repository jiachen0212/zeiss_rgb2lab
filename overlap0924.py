# coding=utf-8
import glob
import os
import cv2
import json
import numpy as np
import xlrd


def imread(path):
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def find_areas(img, color_lower1, color_lower2, color_upper, area_threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, color_lower1, color_upper)
    mask2 = cv2.inRange(hsv, color_lower2, color_upper)
    mask1[:600] = 0
    mask1[-600:] = 0
    mask1[:, :600] = 0
    mask1[:, -600:] = 0
    mask2[:600] = 0
    mask2[-600:] = 0
    mask2[:, :600] = 0
    mask2[:, -600:] = 0
    colors = hsv[:, :, 2][mask1 != 0]
    color = np.percentile(colors, 90) - 20
    mask1 = (hsv[:, :, 2] > color).astype(np.uint8) * mask1

    # cv2.imshow("mask1", mask1)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # cv2.imshow("mask2", mask2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas1 = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        areas1.append(cnt)

    contours, _ = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas2 = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        areas2.append(cnt)

    return areas1, areas2


def cal_color(img, mask):
    color = np.median(img[mask != 0], axis=0)
    return color


def pipeline(img, color_lower1, color_lower2, color_upper, area_threshold):
    draw = img.copy()
    areas1, areas2 = find_areas(img, color_lower1, color_lower2, color_upper, area_threshold)
    if len(areas1) == 0 or len(areas2) == 0:  # 没找到满足条件的area，返回1
        return None, None, 1, draw
    if len(areas1) > 1 or len(areas2) > 1:  # 找到多个满足条件的area，返回2
        return None, None, 2, draw

    mask1 = np.zeros(img.shape[:2], np.float32)
    mask2 = np.zeros(img.shape[:2], np.float32)
    cv2.drawContours(mask1, areas1, 0, 1, -1)
    cv2.drawContours(mask2, areas2, 0, 1, -1)
    mask2 = mask2 - cv2.dilate(mask1, np.ones((7, 7), np.uint8))
    mask2[mask2 < 0] = 0

    color1 = cal_color(img, mask1)
    color2 = cal_color(img, mask2)
    color = color1 - color2

    cv2.drawContours(draw, areas1, 0, (0, 0, 255), 1)
    cv2.drawContours(draw, areas2, 0, (255, 0, 0), 1)
    cv2.putText(draw, "color1: {}".format(color1), (100, 100), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)
    cv2.putText(draw, "color2: {}".format(color2), (100, 200), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)
    cv2.putText(draw, "color: {}".format(color), (100, 300), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)

    return color, "", 0, draw


def main():
    dir_color = dict()
    dirs = [i for i in range(1, 20)]
    for dir in dirs:
        directory = r"D:\work\project\卡尔蔡司AR镀膜\poc\20210924\20210924\{}".format(dir)
        config_dir = r'D:\work\project\卡尔蔡司AR镀膜\poc\20210924_0927_updata_config\20210924\{}'.format(dir)
        paths = glob.glob(os.path.join(directory, "*.bmp"))
        config_file = os.path.join(config_dir, "config.json")
        with open(config_file) as f:
            conf = json.load(f)
            color_lower1 = tuple(conf.get("color_lower1"))
            color_lower2 = tuple(conf.get("color_lower2"))
            color_upper = tuple(conf.get("color_upper"))
            area_threshold = conf.get("area_threshold")
        for path in paths:
            im_name = int(path.split('\\')[-1][:-4])
            img = imread(path)
            color, string, sig, draw = pipeline(img, color_lower1, color_lower2, color_upper, area_threshold)
            print(color)
            dir_color["{}_{}".format(dir + 50, im_name)] = color.tolist()
    data = json.dumps(dir_color)
    with open('./3float_rgb_0924.json', 'w') as js_file:
        js_file.write(data)

    blue = []
    green = []

    for k, v in dir_color.items():
        if v[2] > v[1]:
            blue.append(k)
            print(v)
        else:
            green.append(k)

    return blue, green


def generate_y(base_dir, base_index_value):
    file = os.path.join(base_dir, r'2021-09-23.xlsx')
    wb = xlrd.open_workbook(os.path.join(base_dir, file))
    lab_dict = dict()
    xyz_dict = dict()
    dirs = [i for i in range(1, 20)]
    for i, dir_ind in enumerate(dirs):
        data = wb.sheet_by_name('Sheet{}'.format(dir_ind))
        rows = data.nrows
        # 数据数量check
        assert len(os.listdir(os.path.join(base_dir, str(dir_ind)))) == rows
        title = data.row_values(0)
        l_ind, a_ind, b_ind = title.index('L*'), title.index('a*'), title.index('b*')
        X_ind, Y_ind, Z_ind = title.index('X'), title.index('Y'), title.index('Z')
        for j in range(1, rows):
            l, a, b = data.cell(j, l_ind).value, data.cell(j, a_ind).value, data.cell(j, b_ind).value
            x, y, z = data.cell(j, X_ind).value, data.cell(j, Y_ind).value, data.cell(j, Z_ind).value
            lab_dict["{}_{}".format(dir_ind + base_index_value, j)] = [l, a, b]
            xyz_dict["{}_{}".format(dir_ind + base_index_value, j)] = [x, y, z]
    data1 = json.dumps(lab_dict)
    with open('./lab_value_0924.json', 'w') as js_file:
        js_file.write(data1)
    data2 = json.dumps(xyz_dict)
    with open('./xyz_value_0924.json', 'w') as js_file:
        js_file.write(data2)


def get_pre_dir(blue_dir_name):
    tmp = []
    for b_dir in blue_dir_name:
        a = b_dir.split('_')[0]
        if a not in tmp:
            tmp.append(a)

    return tmp

def split_blue_and_green(blue_dir_name, green_dir_name):
    color = json.load(open('./3float_rgb_0924.json', 'r'))
    lab = json.load(open('./lab_value_0924.json', 'r'))
    assert len(color) == len(lab)

    # for blue
    blue_color = dict()
    blue_lab = dict()
    blue_dir = get_pre_dir(blue_dir_name)
    blue_dir.remove('67')

    # for green
    green_color = dict()
    green_lab = dict()
    green_dir = get_pre_dir(green_dir_name)
    green_dir.remove('65')
    green_dir.remove('68')


    for k, v in color.items():
        if k.split('_')[0] in blue_dir and k in blue_dir_name:
            blue_color[k] = v
            blue_lab[k] = lab[k]
        elif k.split('_')[0] in green_dir and k in green_dir_name:
            green_color[k] = v
            green_lab[k] = lab[k]

    # save json
    data = json.dumps(blue_color)
    with open('./blue_color.json', 'w') as js_file:
        js_file.write(data)
    data = json.dumps(blue_lab)
    with open('./blue_lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(green_color)
    with open('./green_color.json', 'w') as js_file:
        js_file.write(data)
    data = json.dumps(green_lab)
    with open('./green_lab.json', 'w') as js_file:
        js_file.write(data)



if __name__ == '__main__':
    blue_dir_name, green_dir_name = main()
    base_dir = r'D:\work\project\卡尔蔡司AR镀膜\poc\20210924\20210924'
    base_index_value = 50
    # 生成所有文件夹的lab值json
    generate_y(base_dir, base_index_value)
    # 根据 blue_dir_name, green_dir_name 区分蓝绿数据
    split_blue_and_green(blue_dir_name, green_dir_name)
