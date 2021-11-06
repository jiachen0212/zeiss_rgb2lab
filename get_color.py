"""
FILENAME:        get_color.py

AUTHORS:         MoYu

START DATE:      2021.11.05

CONTACT:         yu.mo@smartmore.com

Description:
"""
import os
import json
import glob
import sys
import cv2
import numpy as np


name11 = name12 = name21 = name22 = name31 = name32 = None
area_name = "AREA"
erode_name = "ERODE"
value = ((0, 0, 0), (255, 255, 255))
img = real_img = total_img = np.zeros(0)
area_threshold = 0
erode_threshold = 20
path = color_mode = ""


def imread(p):
    global img
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def change_mode(img, mode):
    if mode == "rgb":
        return img
    if mode == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img


def update(x):
    global value, name11, name12, name21, name22, name31, name32, area_name
    global img, real_img, total_img
    global area_threshold, erode_threshold

    global img, real_img

    v11 = cv2.getTrackbarPos(name11, 'image_win')
    v12 = cv2.getTrackbarPos(name12, 'image_win')
    v21 = cv2.getTrackbarPos(name21, 'image_win')
    v22 = cv2.getTrackbarPos(name22, 'image_win')
    v31 = cv2.getTrackbarPos(name31, 'image_win')
    v32 = cv2.getTrackbarPos(name32, 'image_win')
    area_threshold = cv2.getTrackbarPos(area_name, 'image_win')
    area_threshold = 10 ** (area_threshold / 10)
    erode_threshold = cv2.getTrackbarPos(erode_name, 'image_win')

    value = ((v11, v21, v31), (v12, v22, v32))

    mask = cv2.inRange(real_img, value[0], value[1])
    contours, hie = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    best_idx = -1
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        if area > best_area:
            best_area = area
            best_idx += 1
        areas.append(cnt)

    if areas:
        areas = [areas[best_idx]]

    mask[:] = 0
    for area in areas:
        cv2.drawContours(mask, [area], 0, 255, -1)

    mask = cv2.erode(mask, np.ones((erode_threshold, erode_threshold), np.uint8))

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    total_img[:, img.shape[1] // 4:] = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4))
    cv2.imshow('image_win', total_img)


def train(conf_path):
    global value, name11, name12, name21, name22, name31, name32, area_name, erode_name
    global img, real_img, total_img, path, color_mode

    total_img = np.zeros((img.shape[0] // 4, img.shape[1] // 2, 3), np.uint8)
    total_img[:, :img.shape[1] // 4] = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    cv2.namedWindow('image_win')
    # 手动调整阈值, 然后小白级别得到config..  算法工具化, 便于使用
    cv2.createTrackbar(name11, 'image_win', 0, 255, update)
    cv2.createTrackbar(name12, 'image_win', 0, 255, update)
    cv2.createTrackbar(name21, 'image_win', 0, 255, update)
    cv2.createTrackbar(name22, 'image_win', 0, 255, update)
    cv2.createTrackbar(name31, 'image_win', 0, 255, update)
    cv2.createTrackbar(name32, 'image_win', 0, 255, update)
    cv2.createTrackbar(area_name, 'image_win', 0, 50, update)
    cv2.createTrackbar(erode_name, 'image_win', 0, 200, update)

    cv2.setTrackbarPos(name11, 'image_win', 0)
    cv2.setTrackbarPos(name12, 'image_win', 255)
    cv2.setTrackbarPos(name21, 'image_win', 0)
    cv2.setTrackbarPos(name22, 'image_win', 255)
    cv2.setTrackbarPos(name31, 'image_win', 0)
    cv2.setTrackbarPos(name32, 'image_win', 255)
    cv2.setTrackbarPos(area_name, 'image_win', 30)
    cv2.setTrackbarPos(erode_name, 'image_win', 30)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    conf = {
        "color_range": value,
        "area_threshold": area_threshold,
        "erode_threshold": erode_threshold
    }
    with open(conf_path, "w") as f:
        json.dump(conf, f)


def test(conf_path):
    global img, real_img, total_img, path
    global value, area_threshold, erode_threshold
    with open(conf_path) as f:
        conf = json.load(f)
        value = conf.get("color_range")
        area_threshold = conf.get("area_threshold")
        erode_threshold = conf.get("erode_threshold")

    image_paths = glob.glob(os.path.join(path, "*.bmp"))
    for p in image_paths:
        img = imread(p)
        real_img = change_mode(img, color_mode)
        mask = cv2.inRange(real_img, tuple(value[0]), tuple(value[1]))
        contours, hie = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        best_idx = -1
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                continue
            if area > best_area:
                best_area = area
                best_idx += 1
            areas.append(cnt)

        if areas:
            areas = [areas[best_idx]]

        mask[:] = 0
        for area in areas:
            cv2.drawContours(mask, [area], 0, 255, -1)

        if mask.sum() == 0:
            print(p, "bad result")
            continue

        color = img[mask != 0].mean(axis=0)
        color = [round(c, 2) for c in color]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.putText(img, "Color: {}".format(color), (100, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)
        total_img = np.zeros((img.shape[0] // 4, img.shape[1] // 2, 3), np.uint8)
        total_img[:, :img.shape[1] // 4] = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        total_img[:, img.shape[1] // 4:] = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4))
        cv2.imshow('image_win', total_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(p, color)



def main():
    if len(sys.argv) != 5:
        sys.exit("python get_color.py [rgb/hsv] [train/test] [path] [config_path]")

    global color_mode, path, img, real_img
    color_mode, process_mode, path, conf_path = sys.argv[1:]
    print(color_mode, process_mode, path, conf_path)

    global value, name11, name12, name21, name22, name31, name32

    if color_mode == "rgb":
        name11 = "R LOWER"
        name12 = "R UPPER"
        name21 = "G LOWER"
        name22 = "G UPPER"
        name31 = "B LOWER"
        name32 = "B UPPER"
    else:
        name11 = "H LOWER"
        name12 = "H UPPER"
        name21 = "S LOWER"
        name22 = "S UPPER"
        name31 = "V LOWER"
        name32 = "V UPPER"

    if process_mode == "train":
        img = imread(path)
        real_img = change_mode(img, color_mode)
        train(conf_path)
    else:
        test(conf_path)


if __name__ == '__main__':
    main()
