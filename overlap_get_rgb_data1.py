# coding=utf-8
import os
import cv2
import numpy as np
import json

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


def slim_roi(area, img):
    # 基于选中的蓝色roi, 选择roi的相对中心的小框  [实验下来, 貌似效果不如取全部roi的均值]
    area = area.squeeze(axis=1)
    xs, ys = [], []
    for tt in area:
        xs.append(tt[0])
        ys.append(tt[1])
    x_min, x_max, y_min, y_max = int((min(xs)+max(xs))/2) - 50, int((min(xs)+max(xs))/2) + 50, int((min(ys) + max(ys))/2) - 50, int((min(ys) + max(ys))/2) + 50
    p1, p2, p3, p4 = [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
    mask1 = np.zeros((img.shape), dtype=np.uint8)
    pts = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    cv2.fillPoly(mask1, pts, (255, 255, 255))
    values = img[np.where((mask1 == (255, 255, 255)).all(axis=2))].mean(axis=0)
    cv2.rectangle(img, p4, p2, (0, 255, 255), 3)
    cv2.drawContours(img, [area], 0, (255, 0, 0), 2)
    cv2.imwrite('./diamond_mask.png', img)
    print("values: {}".format(values))

    return values



def show_area(mask):
    # mask!=0的所有pixel会被提取. 不是规整的矩形区域..
    points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                points.append([i, j])
    for point in points[:500]:
        plt.scatter(point[0], point[1], marker = 'x',color = 'red', s = 40 ,label = 'First')
    plt.show()


import collections
def get_distribute(r_, topk):
    distribute = collections.Counter(r_)
    # 防止设置的topk超过分布数目
    topk = min(topk, len(distribute))
    # show_distribute(r_)
    sored = sorted(distribute.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
    sum_color = 0
    count = 0
    for i in range(topk):
        sum_color += sored[i][0] * sored[i][1]
        count += sored[i][1]

    return float(sum_color / count)


def slim_roi_rgb_distracte(img, mask):
    tmp = img[mask != 0]
    # 保留出现次数的topk像素, 丢弃其他, 然后这个部分取均值
    topk = 6
    # # 丢弃分布中看两边k个数据, 剩下ll-2*k 取颜色均值
    # # remove_k = 2
    filtered_r = get_distribute(tmp[:, 0], topk)
    filtered_g = get_distribute(tmp[:, 1], topk)
    filtered_b = get_distribute(tmp[:, 2], topk)

    return [filtered_r, filtered_g, filtered_b]


import matplotlib.pyplot as plt
def cal_color(img, area, im_name):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [area], 0, 1, -1)

    # print(img[mask != 0].shape)

    # mask区域向内缩180个像素
    mask = cv2.erode(mask, np.ones((180, 180), np.uint8))

    # print(img[mask != 0].shape)
    # tmp = np.nonzero(mask)
    # top_left, bottom_right = [min(tmp[0]), min(tmp[1])], [max(tmp[0]), max(tmp[1])]
    # cv2.rectangle(img, top_left, bottom_right, (255, 0, 0))
    # cv2.imwrite('{}.png'.format(im_name), img)
    
    color = img[mask != 0].mean(axis=0)

    # color = slim_roi(area, img)

    # 精简roi的分布, 保留中心分布像素值们的均值
    # color = slim_roi_rgb_distracte(img, mask)
    # return color.astype(np.uint8)
    return color


def is_ok(color, color_thresholds):
    r, g, b = color
    (r1, g1, b1), (r2, g2, b2) = color_thresholds

    return r1 <= r <= r2 and g1 <= g <= g2 and b1 <= b <= b2


def pipeline(img, color_lower, color_upper, area_threshold, flag, im_name):
    areas = find_areas(img, color_lower, color_upper, area_threshold)
    if len(areas) == 0:  # 没找到满足条件的area，返回1
        return None, None, 1, None
    if len(areas) > 1:  # 找到多个满足条件的area，返回2
        return None, None, 2, None

    # 1102
    # #1. 非质心法
    if not flag:
        color = cal_color(img, areas[0], im_name)
    else:
        # 2. 质心法
        moments = cv2.moments(areas[0])
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        # print("rio中心点坐标: ({}, {})".format(cx, cy))
        half_height = 88
        half_width = 88
        area = np.array([
            [cx - half_width, cy - half_height],
            [cx - half_width, cy + half_height],
            [cx + half_width, cy + half_height],
            [cx + half_width, cy - half_height]
        ], np.int32)
        # cv2.rectangle(img, area[0], area[2], (255, 0, 0))
        # cv2.imwrite('./{}.png'.format(im_name), img)
        color = cal_color(img, area, im_name)

    return color


def main(single_dir_col, dir_index, path, ff, area_thresholds, flag):
    import glob
    paths = glob.glob(os.path.join(path, r"*.bmp").format(path))
    for area_threshold in area_thresholds:
        print("area_threshold : {}".format(area_threshold))
        for ind, path in enumerate(paths):
            print(path)
            ff.write(path+'\n')
            im_name = int(path.split('\\')[-1][:-4])
            img = imread(path)
            rect = np.array([[1180, 1100], [1230, 1100], [1230, 1150], [1180, 1150]])   

            color_lower, color_upper = cal_color_range(img, rect)   
            # color_lower = (0, 120, 0)
            # color_upper = (200, 200, 200)

            # 1019
            # dir1
            # color_lower = (20, 45, 40)
            # color_upper = (40, 75, 70)
            # dir2 dir3
            color_lower = (30, 110, 80)
            color_upper = (70, 160, 140)

            # area_threshold = 1000
            # color_thresholds = ((0, 0, 0), (255, 255, 255))  # 用户设定的颜色阈值, 用于ok/ng
            color = pipeline(img, color_lower, color_upper, area_threshold, flag, im_name)
            print(color)
            ff.write("color: " + ''.join(str(a)+',   ' for a in color) + '\n')

            # single_dir_col["{}_{}".format(dir_index, im_name)] = [str(a) for a in color]
        ff.write('\n\n')


if __name__ == '__main__':

    dir_n = 3
    save_json_dir = r'D:\work\project\卡尔蔡司膜色缺陷\data'
    base_dir = r'C:\Users\15974\Desktop\20211102新镜片\20211102新镜片'
    # area_thresholds = [1000, 2500, 1500, 2000, 3000, 3500, 10000, 15000, 20000, 25000, 30000]
    area_thresholds = [60000]
    flag = 0

    all_col3 = dict()
    ff = open(r'./Get_RGB_Value.txt', 'a')
    for i in range(3, dir_n+1):
        dir_path = os.path.join(base_dir, str(i))
        main(all_col3, i, dir_path, ff, area_thresholds, flag)
    # data = json.dumps(all_col3)
    # with open(os.path.join(save_json_dir, 'data1_rgb.json'), 'w') as js_file:
    #     js_file.write(data)

