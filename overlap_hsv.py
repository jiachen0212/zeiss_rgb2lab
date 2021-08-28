"""
FILENAME:        overlap_hsv.py

AUTHORS:         MoYu

START DATE:      2021.08.10

CONTACT:         yu.mo@smartmore.com

Description:
"""


import cv2
import numpy as np


def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def find_areas(img, color_lower, color_upper, area_threshold):
    mask1 = (img > color_upper).astype(np.uint8) * 255
    mask2 = (img > color_lower).astype(np.uint8) * 255

    area_cnt1 = area_cnt2 = None
    best_area = 0

    contours, hie = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        if best_area > area:
            continue

        area_cnt1 = cnt
        best_area = area

    best_area = 0
    contours, hie = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        if best_area > area:
            continue

        area_cnt2 = cnt
        best_area = area

    return area_cnt1, area_cnt2


def pipeline(img, color_lower, color_upper, area_threshold):
    draw = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]

    area_cnt1, area_cnt2 = find_areas(v_channel, color_lower, color_upper, area_threshold)
    if area_cnt1 is None or area_cnt2 is None:
        return None, None, draw

    mask1 = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask1, [area_cnt1], 0, 255, -1)
    mask2 = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask2, [area_cnt2], 0, 255, -1)
    mask2 = mask2 - mask1

    color1 = img[mask1 != 0].mean(axis=0).astype(np.uint8)
    color2 = img[mask2 != 0].mean(axis=0).astype(np.uint8)

    cv2.drawContours(draw, [area_cnt1], 0, (255, 0, 0), 2)
    cv2.drawContours(draw, [area_cnt2], 0, (0, 0, 255), 2)
    cv2.putText(draw, "color1: {}".format(color1), (100, 100), cv2.FONT_ITALIC, 3, (255, 0, 0), 2)
    cv2.putText(draw, "color2: {}".format(color2), (100, 250), cv2.FONT_ITALIC, 3, (0, 0, 255), 2)
    print(color1, color2, '---')
    return color1, color2, draw


def main(single_dir_col, dir_index, path):
    import glob
    paths = glob.glob(os.path.join(path, r"*.bmp").format(path))
    for ind, path in enumerate(paths):
        print(path)
        img = imread(path)

        color_lower = 75
        color_upper = 120
        area_threshold = 1000  # 用户设定的面积阈值

        color1, color2, draw = pipeline(img, color_lower, color_upper, area_threshold)
        color = []
        for a in color1:
            color.append(a)
        for a in color2:
            color.append(a)

        single_dir_col["{}_{}".format(dir_index, ind + 1)] = [str(a) for a in color]
        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    import os
    import json

    dir_n = 8
    # base_dir = r"D:\work\project\卡尔蔡司AR镀膜\poc\膜色图像数据"
    base_dir = r'D:\work\project\卡尔蔡司AR镀膜\poc\0812\0812'
    all_col6 = dict()
    for i in range(1, dir_n+1):
        dir_path = os.path.join(base_dir, str(i))
        main(all_col6, i+20, dir_path)
    data = json.dumps(all_col6)
    with open('./all_col6_0817.json', 'w') as js_file:
        js_file.write(data)
