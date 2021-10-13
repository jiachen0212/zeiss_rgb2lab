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
    areas = find_areas(img, color_lower, color_upper, area_threshold)
    if len(areas) == 0:  # 没找到满足条件的area，返回1
        return None, None, 1, None
    if len(areas) > 1:  # 找到多个满足条件的area，返回2
        return None, None, 2, None

    color = cal_color(img, areas[0])
    print("overlap color: {}".format(color))
    return color


def main(single_dir_col, dir_index, path):
    import glob
    paths = glob.glob(os.path.join(path, r"*.bmp").format(path))
    for ind, path in enumerate(paths):
        print(path)
        im_name = int(path.split('\\')[-1][:-4])
        img = imread(path)
        rect = np.array([[1180, 1100], [1230, 1100], [1230, 1150], [1180, 1150]])  # 用户画的框

        color_lower, color_upper = cal_color_range(img, rect)  # 根据用户画的框计算出颜色上下界，用于后续区域分割
        color_lower = (0, 120, 0)
        color_upper = (200, 200, 200)
        area_threshold = 1000  # 用户设定的面积阈值
        color_thresholds = ((0, 0, 0), (255, 255, 255))  # 用户设定的颜色阈值, 用于ok/ng

        # color, string, sig, draw = pipeline(img, color_lower, color_upper, area_threshold, color_thresholds)
        # if sig == 0:
        #     cv2.imshow("draw", draw)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        color = pipeline(img, color_lower, color_upper, area_threshold, color_thresholds)
        single_dir_col["{}_{}".format(dir_index, im_name)] = [str(a) for a in color]



if __name__ == '__main__':
    '''
    key的顺序: 1-6 
    
    '''
    dir_n = 6
    save_json_dir = r'D:\work\project\卡尔蔡司膜色缺陷\data'
    base_dir = r"D:\work\project\卡尔蔡司膜色缺陷\data\data1"
    all_col3 = dict()
    for i in range(1, dir_n+1):
        dir_path = os.path.join(base_dir, str(i))
        main(all_col3, i, dir_path)
    data = json.dumps(all_col3)
    with open(os.path.join(save_json_dir, './data1_rgb.json'), 'w') as js_file:
        js_file.write(data)



