# coding=utf-8
import os
import cv2
import numpy as np
import json
import xlrd
import glob


'''
for data0924 blue imgs
生成: './3float_rgb_0924.json'  蓝绿膜数据的 3个 float rgb 值
生成: 蓝绿膜数据的 lab 值:  './lab_value_0924.json'
              xyz 值:  './xyz_value_0924.json'
              
'''


def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def cal_color_range(img, rect):
    top_left = rect[0]
    bottom_right = rect[2]
    x1, y1 = top_left
    x2, y2 = bottom_right

    sub = img[y1:y2, x1:x2]
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    mean_r = round(r.mean())
    mean_g = round(g.mean())
    mean_b = round(b.mean())

    std_r = max(r.std(), 5)
    std_g = max(g.std(), 5)
    std_b = max(b.std(), 5)

    r1, r2 = mean_r - std_r * 2, mean_r + std_r * 2
    g1, g2 = mean_g - std_g * 2, mean_g + std_g * 2
    b1, b2 = mean_b - std_b * 2, mean_b + std_b * 2

    return (r1, g1, b1), (r2, g2, b2)


def find_areas(img, color_lower, color_upper, area_threshold):
    mask = cv2.inRange(img, color_lower, color_upper)
    mask[:200] = 0
    mask[-200:] = 0
    mask[:, :200] = 0
    mask[:, -200:] = 0
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        areas.append(cnt)

    return areas


def cal_color(img, area):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [area], 0, 1, -1)
    color = img[mask != 0].mean(axis=0)
    # return color.astype(np.uint8)
    return color


def is_ok(color, color_thresholds):
    r, g, b = color
    (r1, g1, b1), (r2, g2, b2) = color_thresholds

    return r1 <= r <= r2 and g1 <= g <= g2 and b1 <= b <= b2


def pipeline(img, color_lower, color_upper, area_threshold, color_thresholds):
    draw = img.copy()
    areas = find_areas(img, color_lower, color_upper, area_threshold)
    if len(areas) == 0:  # 没找到满足条件的area，返回1
        return None, None, 1, draw
    if len(areas) > 1:  # 找到多个满足条件的area，返回2
        return None, None, 2, draw

    moments = cv2.moments(areas[0])
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    half_height = 20
    half_width = 20
    area = np.array([
        [cx - half_width, cy - half_height],
        [cx - half_width, cy + half_height],
        [cx + half_width, cy + half_height],
        [cx + half_width, cy - half_height]
    ], np.int32)

    color = cal_color(img, area)
    cv2.drawContours(draw, [area], 0, (0, 0, 255), 2)
    cv2.putText(draw, "color: {}".format(color), (100, 100), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)

    if is_ok(color, color_thresholds):
        cv2.putText(draw, "result: OK", (100, 200), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)
        return color, "OK", 0, draw
    else:
        cv2.putText(draw, "result: NG", (100, 200), cv2.FONT_ITALIC, 3, (0, 255, 255), 2)
        return color, "NG", 0, draw


def main(single_dir_col, dir_index, dir_path, base_index_value, isblue):
    # 等待 mo_yu 哥针对每个文件夹定制的 config.json
    config_file = os.path.join(dir_path, r'D:\work\project\卡尔蔡司AR镀膜\poc\data_0924\1\config_green.json')
    with open(config_file) as f:
        conf = json.load(f)
        color_lower = tuple(conf.get("color_lower"))
        color_upper = tuple(conf.get("color_upper"))
        area_threshold = conf.get("area_threshold")
        color_thresholds = tuple(conf.get("color_thresholds"))

    im_paths = glob.glob(os.path.join(dir_path, r"*.bmp").format(dir_path))
    for ind, im_path in enumerate(im_paths):
        im_name = int(im_path.split('\\')[-1][:-4])
        img = imread(im_path)
        color, string, sig, draw = pipeline(img, color_lower, color_upper, area_threshold, color_thresholds)

        # chenjia debug 等待 mo_yu 哥针对每个文件夹定制的config.json
        try:
            single_dir_col["{}_{}".format(dir_index + base_index_value, im_name)] = [str(a) for a in color]

            # 根据计算到的color值, 区分蓝绿膜: color rgb
            if isblue:
                if color[2] < color[1]:
                    print("color: {}\tblue data, but blue < green..\t{}".format(color, im_path))
            else:
                if color[2] > color[1]:
                    print("color: {}\tgreen data, but blue > green..\t{}".format(color, im_path))
        except:
            color = [33.66983938, 113.96252231, 199.80785247]
            single_dir_col["{}_{}".format(dir_index + base_index_value, im_name)] = [str(a) for a in color]


# def lab2xyz(l,a,b):
#     fy = (l+16.0) / 116.0
#     fx = a / 500.0 + fy
#     fz = fy - b / 200.0
#     if np.power(fy, 3) > 0.008856:
#         y = np.power(fy, 3)
#     else:
#         y = (fy - 16 / 116.0) / 7.787
#
#     if np.power(fx, 3) > 0.008856:
#         x = np.power(fx, 3)
#     else:
#         x = (fx - 16 / 116.0) / 7.787
#
#     if np.power(fz, 3) > 0.008856:
#         z = np.power(fz, 3)
#     else:
#         z = (fz - 16 / 116.0) / 7.787
#     x *= 94.81211415
#     y *= 100
#     z *= 107.3369399
#
#     return [x,y,z]


def generate_x(blue_dir, base_index_value, isblue):
    all_col3 = dict()
    for i, dir_name in enumerate(blue_dir):
        dir_path = os.path.join(base_dir, dir_name)
        main(all_col3, i, dir_path, base_index_value, isblue)
    data = json.dumps(all_col3)
    print(all_col3.keys())
    with open('./3float_rgb_0924.json', 'w') as js_file:
        js_file.write(data)


def generate_y(blue_dir, blue_number, base_index_value):
    file = os.path.join(base_dir, r'2021-09-23.xlsx')
    wb = xlrd.open_workbook(os.path.join(base_dir, file))
    lab_dict = dict()
    xyz_dict = dict()
    for i, dir_ind in enumerate(blue_number):
        data = wb.sheet_by_name('Sheet{}'.format(dir_ind))
        rows = data.nrows
        # 数据数量check
        assert len(os.listdir(os.path.join(base_dir, blue_dir[i]))) == rows - 1
        title = data.row_values(0)
        l_ind, a_ind, b_ind = title.index('L*'), title.index('a*'), title.index('b*')
        X_ind, Y_ind, Z_ind = title.index('X'), title.index('Y'), title.index('Z')
        for j in range(1, rows):
            l, a, b = data.cell(j, l_ind).value, data.cell(j, a_ind).value, data.cell(j, b_ind).value
            x, y, z = data.cell(j, X_ind).value, data.cell(j, Y_ind).value, data.cell(j, Z_ind).value
            # print(l, a, b)
            lab_dict["{}_{}".format(i + base_index_value, j)] = [l, a, b]
            # print(lab2xyz(l, a, b))
            xyz_dict["{}_{}".format(i + base_index_value, j)] = [x, y, z]
            # print(x, y, z)
    print(lab_dict.keys())
    data1 = json.dumps(lab_dict)
    with open('./lab_value_0924.json', 'w') as js_file:
        js_file.write(data1)
    data2 = json.dumps(xyz_dict)
    with open('./xyz_value_0924.json', 'w') as js_file:
        js_file.write(data2)



if __name__ == '__main__':

    base_dir = r"D:\work\project\卡尔蔡司AR镀膜\poc\data_0924\1"

    blue_dir = [r'1（纯蓝）', r'5（纯蓝）', r'7  (纯蓝)', r'8  (纯蓝)', r'9（纯蓝）', r'12（纯蓝）', r'13（纯蓝）', r'14 (纯蓝)', r'16 (纯蓝)', r'19 纯蓝']
    blue_number = [1, 5, 7, 8, 9, 12, 13, 14, 16, 19]

    green_dir = ['3（纯绿）', '10（纯绿）', '11（纯绿）']
    green_number = [3, 10, 11]

    # 针对混合的数据文件夹, 拆分开蓝绿数据.
    mixup_dir = [r'2', r'4', r'6']
    mixup_number = [2, 4, 6]

    isblue = 0

    # isblue: 0green 1blue
    if isblue:
        base_index_value = 50
    else:
        base_index_value = 100

    # step1. for float rgb value
    generate_x(green_dir, base_index_value, isblue)
    # # step2. for lab value
    # generate_y(green_dir, green_number, base_index_value)
