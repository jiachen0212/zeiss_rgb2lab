# coding=utf-8
import glob
import os
import cv2
import json
import numpy as np
import xlrd


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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask[:600] = 0
    mask[-600:] = 0
    mask[:, :600] = 0
    mask[:, -600:] = 0
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    colors = hsv[:, :, 2][mask != 0]
    color = np.percentile(colors, 90) - 20
    mask = (hsv[:, :, 2] > color).astype(np.uint8) * mask
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
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


def main():
    dir_color = dict()
    dirs = [i for i in range(1, 20)]
    for dir in dirs:
        directory = r"D:\work\project\卡尔蔡司AR镀膜\poc\20210924\20210924\{}".format(dir)
        paths = glob.glob(os.path.join(directory, "*.bmp"))
        config_file = os.path.join(directory, "config.json")
        with open(config_file) as f:
            conf = json.load(f)
            color_lower = tuple(conf.get("color_lower"))
            color_upper = tuple(conf.get("color_upper"))
            area_threshold = conf.get("area_threshold")
            color_thresholds = tuple(conf.get("color_thresholds"))

        for path in paths:
            im_name = int(path.split('\\')[-1][:-4])
            # print(im_name)
            # print("{}_{}".format(dir + 50, im_name))
            img = imread(path)
            color, string, sig, draw = pipeline(img, color_lower, color_upper, area_threshold, color_thresholds)
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
