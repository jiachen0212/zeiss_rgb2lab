# coding=utf-8
# FILENAME:        get_color.py
#
# AUTHORS:         MoYu
#
# START DATE:      2021.11.05
#
# CONTACT:         yu.mo@smartmore.com
#
# Description:
# python get_color.py "rgb" "train" "./1118data/1118/16/14.bmp" "./1118data/1118/16.json"
# python get_color.py "rgb" "test" "./1118data/膜色1118/1118/1" "./1118data/膜色1118/1118/1.json"

import os
import json
import glob
import sys
import cv2
import xlrd
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
            best_idx = len(areas)
        areas.append(cnt)

    mask[:] = 0
    for area in areas:
        cv2.drawContours(mask, [area], 0, 255, -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask, areas, best_idx, (0, 0, 255), -1)
    mask = cv2.erode(mask, np.ones((erode_threshold, erode_threshold), np.uint8))
    img1 = img.copy()

    img1[mask != 0] = 255

    total_img[:, img.shape[1] // 4:] = cv2.resize(img1, (mask.shape[1] // 4, mask.shape[0] // 4))  # cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4))
    cv2.imshow('image_win', total_img)


def train(conf_path):
    global value, name11, name12, name21, name22, name31, name32, area_name, erode_name
    global img, real_img, total_img, path, color_mode

    total_img = np.zeros((img.shape[0] // 4, img.shape[1] // 2, 3), np.uint8)
    total_img[:, :img.shape[1] // 4] = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    cv2.namedWindow('image_win')
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


import collections
import matplotlib.pyplot as plt
def show_distribute(r_, color):
    plt.hist(x=r_, bins='auto', color=color,
             alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

def get_distribute(r_, topk, color=None):
    distribute = collections.Counter(r_)
    # 防止设置的topk超过分布数目
    topk = min(topk, len(distribute))
    # topk = len(distribute)
    show_distribute(r_, color)
    sored = sorted(distribute.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
    sum_color = 0
    count = 0
    for i in range(topk):
        sum_color += sored[i][0] * sored[i][1]
        count += sored[i][1]

    return float(sum_color / count)

def slim_roi_rgb_distracte(img, mask, p):
    # print(img[mask != 0].mean(axis=0), '8')
    tmp = img[mask != 0]
    # 保留出现次数的topk像素, 丢弃其他, 然后这个部分取均值
    topk = 4
    # # 丢弃分布中看两边k个数据, 剩下ll-2*k 取颜色均值
    # # remove_k = 2
    filtered_r = get_distribute(tmp[:, 0], topk, color='red')
    filtered_g = get_distribute(tmp[:, 1], topk, color='green')
    filtered_b = get_distribute(tmp[:, 2], topk, color='blue')
    # plt.show()
    # 保存每一张img的roi RGB值分布情况
    plt.savefig(p[:-4] + '_roi.png')

    return [filtered_r, filtered_g, filtered_b]


def test(conf_path):
    global img, real_img, total_img, path
    global value, area_threshold, erode_threshold
    with open(conf_path) as f:
        conf = json.load(f)
        value = conf.get("color_range")
        area_threshold = conf.get("area_threshold")
        erode_threshold = conf.get("erode_threshold")

    image_paths = glob.glob(os.path.join(path, "*.bmp"))
    dir_color = dict()
    for p in image_paths:
        img = imread(p)
        real_img = change_mode(img, color_mode)
        mask = cv2.inRange(real_img, tuple(value[0]), tuple(value[1]))
        contours, hie = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_threshold:
                continue
            if area > best_area:
                best_area = area
                areas = [cnt]

        mask[:] = 0
        for area in areas:
            cv2.drawContours(mask, [area], 0, 255, -1)

        mask = cv2.erode(mask, np.ones((erode_threshold, erode_threshold), np.uint8))
        mask_area = mask.sum()
        if mask_area == 0:
            print(p, "bad result")
            continue

        color = img[mask != 0].mean(axis=0)
        # color = slim_roi_rgb_distracte(img, mask, p)

        color = [round(c, 2) for c in color]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img1 = img.copy()
        img1[mask != 0] = 255

        tmp1, tmp2 = p.split('/')[-2], p.split('/')[-1]
        img_name = "{}_{}".format(tmp1, tmp2.split('.')[0])
        print("img_name: {}".format(img_name))
        cv2.putText(img, "Color: {}， img_name: {}".format(color, img_name), (100, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)
        total_img = np.zeros((img.shape[0] // 4, img.shape[1] // 2, 3), np.uint8)
        total_img[:, :img.shape[1] // 4] = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        total_img[:, img.shape[1] // 4:] = cv2.resize(img1, (mask.shape[1] // 4, mask.shape[0] // 4))
        cv2.imshow('image_win', total_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        dir_color[img_name] = color + [str(mask_area)]
        print(img_name, dir_color[img_name])
        data = json.dumps(dir_color)
        with open(r'./1122_rgb_js/dir_{}_rgb.json'.format(tmp1), 'w') as js_file:
            js_file.write(data)


def main():
    if len(sys.argv) != 5:
        sys.exit("python get_color.py [rgb/hsv] [train/test] [path] [config_path]")

    global color_mode, path, img, real_img
    color_mode, process_mode, path, conf_path = sys.argv[1:]

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



def get_lab():
    # green data, g < b
    bad_green = ['5_9', '5_13', '5_10', '15_13']
    L = [9, 14]
    A = [-24, -15]
    B = [-2, 10]
    zc, yc = dict(), dict()
    lab_file = r'./1118data/膜色识别11-18.xlsx'
    wb = xlrd.open_workbook(lab_file)
    all_lab_value = dict()
    all_pass_no_pass = dict()
    ind1 = [1, 2, 3, 7, 8, 9, 10, 11, 14, 15, 16]
    ind2 = [4, 5, 6, 12, 13]
    sheets = ["Sheet{}".format(ind) for ind in ind1]
    sheets += ["Sheet{}+".format(ind) for ind in ind2]
    test_ims = []
    for sheet in sheets:
        data = wb.sheet_by_name(sheet)
        rows = data.nrows
        title = data.row_values(0)
        l_index, im_name_index, pass_index = title.index("L*"), title.index("ID"), title.index('通过/不通过')
        for j in range(1, rows):
            row_data = data.row_values(j)
            img_name = row_data[im_name_index]
            img_name = "{}_{}".format(img_name.split('-')[0], img_name.split('-')[1])
            l, a, b = row_data[l_index], row_data[l_index + 1], row_data[l_index + 2]
            try:
                float(l)
            except:
                test_ims.append(img_name)
                continue
            all_lab_value[img_name] = [l, a, b]
            all_pass_no_pass[img_name] = row_data[pass_index]
            # 统计样本的正常异常情况
            if (L[0] <= float(l) <= L[1]) and (A[0] <= float(a) <= A[1]) and (B[0] <= float(b) <= B[1]):
                zc[img_name] = [l,a,b]
            else:
                yc[img_name] = [l,a,b]
    print("len zc: {}".format(len(zc)))
    print("len yc: {}, {}".format(len(yc), yc))

    # 漏掉了这张bmp
    del all_lab_value['16_14']
    print("剔除异常样本前, size: ", len(all_lab_value))
    # 剔除异常样本
    all_im_names = list(all_lab_value.keys())
    for k in all_im_names:
        if (k in yc) or (k in bad_green) or (k.split('_')[0] in ['15', '16']):
            del all_lab_value[k]
    print("剔除异常样本后, size: ", len(all_lab_value))

    data = json.dumps(all_lab_value)
    with open(r'./1118_lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(all_pass_no_pass)
    with open(r'./1118_pass_no_pass.json', 'w') as js_file:
        js_file.write(data)

    ff = open('./test_ims.txt', 'w')
    for tes in test_ims:
        ff.write(tes + ',')
    ff.close()

    # split train and test
    inds = [i for i in range(1, 17)]
    all_trian_rgb = dict()
    all_test_rgb = dict()
    yc_rgb = dict()
    for ind in inds:
        js = json.load(open(r'./1122_rgb_js/dir_{}_rgb.json'.format(ind), 'r'))
        for k, v in js.items():
            if (k not in test_ims) and (k not in yc) and (k not in bad_green) and ((k.split('_')[0] not in ['15', '16'])):
                all_trian_rgb[k] = v
            elif (k not in yc) and (k not in bad_green) and ((k.split('_')[0] not in ['15', '16'])):
                all_test_rgb[k] = v
            elif k in yc:
                yc_rgb[k] = v
    data = json.dumps(all_trian_rgb)
    with open(r'./1118_train_rgb.json', 'w') as js_file:
        js_file.write(data)
    print(len(all_trian_rgb), len(all_test_rgb), len(all_lab_value))
    assert len(all_trian_rgb) == len(all_lab_value)
    print("train data: {}".format(len(all_trian_rgb)))
    data = json.dumps(all_test_rgb)
    with open(r'./1118_test_rgb.json', 'w') as js_file:
        js_file.write(data)
    print("test data: {}".format(len(all_test_rgb)))

    # LAB异常样本落盘
    data = json.dumps(yc)
    with open(r'./bad_lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(yc_rgb)
    with open(r'./bad_rgb.json', 'w') as js_file:
        js_file.write(data)



def get_lab1(test_dir):
    '''
    留一个文件夹不训 做测试数据
    '''
    bad_green = ['5_9', '5_13', '5_10', '15_13']
    L = [9, 14]
    A = [-24, -15]
    B = [-2, 10]
    zc, yc = dict(), dict()
    lab_file = r'./1118data/膜色识别11-18.xlsx'
    wb = xlrd.open_workbook(lab_file)
    all_lab_value = dict()
    all_pass_no_pass = dict()
    ind1 = [1, 2, 3, 7, 8, 9, 10, 11, 14, 15, 16]
    ind2 = [4, 5, 6, 12, 13]
    sheets = ["Sheet{}".format(ind) for ind in ind1]
    sheets += ["Sheet{}+".format(ind) for ind in ind2]
    test_ims = []
    for sheet in sheets:
        data = wb.sheet_by_name(sheet)
        rows = data.nrows
        title = data.row_values(0)
        l_index, im_name_index, pass_index = title.index("L*"), title.index("ID"), title.index('通过/不通过')
        for j in range(1, rows):
            row_data = data.row_values(j)
            img_name = row_data[im_name_index]
            img_name = "{}_{}".format(img_name.split('-')[0], img_name.split('-')[1])
            l, a, b = row_data[l_index], row_data[l_index + 1], row_data[l_index + 2]
            try:
                float(l)
            except:
                test_ims.append(img_name)
                continue
            all_lab_value[img_name] = [l, a, b]
            all_pass_no_pass[img_name] = row_data[pass_index]
            # 统计样本的正常异常情况
            if (L[0] <= float(l) <= L[1]) and (A[0] <= float(a) <= A[1]) and (B[0] <= float(b) <= B[1]):
                zc[img_name] = [l,a,b]
            else:
                yc[img_name] = [l,a,b]
    del all_lab_value['16_14']

    test_lab = dict()
    all_im_names = list(all_lab_value.keys())
    for k in all_im_names:
        if k.split('_')[0] == test_dir:
            test_lab[k] = all_lab_value[k]
        if (k in yc) or (k in bad_green) or (k.split('_')[0] in ['15', '16', test_dir]):
            del all_lab_value[k]

    data = json.dumps(all_lab_value)
    with open(r'./1118_lab.json', 'w') as js_file:
        js_file.write(data)

    # 落盘test_dir的 gt_lab
    data = json.dumps(test_lab)
    with open(r'./{}_lab.json'.format(test_dir), 'w') as js_file:
        js_file.write(data)

    # split train and test rgb
    inds = [i for i in range(1, 17)]
    all_trian_rgb = dict()
    all_test_rgb = dict()
    yc_rgb = dict()
    for ind in inds:
        js = json.load(open(r'./1122_rgb_js/dir_{}_rgb.json'.format(ind), 'r'))
        for k, v in js.items():
            if (k not in test_ims) and (k not in yc) and (k not in bad_green) and ((k.split('_')[0] not in ['15', '16', test_dir])):
                all_trian_rgb[k] = v
            elif (k not in yc) and (k not in bad_green) and ((k.split('_')[0] not in ['15', '16'])):
                if k.split('_')[0] == test_dir:
                    all_test_rgb[k] = v
            elif k in yc:
                yc_rgb[k] = v

    data = json.dumps(all_trian_rgb)
    with open(r'./1118_train_rgb.json', 'w') as js_file:
        js_file.write(data)
    print(len(all_trian_rgb), len(all_test_rgb), len(all_lab_value))
    assert len(all_trian_rgb) == len(all_lab_value)
    print("train data: {}".format(len(all_trian_rgb)))

    # 剔除test中没有lab值的几条样本
    del all_test_rgb['8_7']
    del all_test_rgb['8_20']

    data = json.dumps(all_test_rgb)
    with open(r'./1118_test_rgb.json', 'w') as js_file:
        js_file.write(data)
    print("test data: {}".format(len(all_test_rgb)))

    # LAB异常样本落盘
    data = json.dumps(yc)
    with open(r'./bad_lab.json', 'w') as js_file:
        js_file.write(data)
    data = json.dumps(yc_rgb)
    with open(r'./bad_rgb.json', 'w') as js_file:
        js_file.write(data)


def show_dir_ng_ok():
    all_ok, all_ng = 0, 0
    ind1 = [1, 2, 3, 7, 8, 9, 10, 11, 14, 15, 16]
    ind2 = [4, 5, 6, 12, 13]
    sheets = ["Sheet{}".format(ind) for ind in ind1]
    sheets += ["Sheet{}+".format(ind) for ind in ind2]
    lab_file = r'./1118data/膜色识别11-18.xlsx'
    wb = xlrd.open_workbook(lab_file)
    for sheet in sheets:
        data = wb.sheet_by_name(sheet)
        rows = data.nrows
        index = data.row_values(0).index('通过/不通过')
        ok, ng = 0, 0
        for j in range(1, rows):
            row_data = data.row_values(j)
            is_no_pass = row_data[index]
            if len(is_no_pass) == 3:
                ng += 1
            elif len(is_no_pass) == 2:
                ok += 1
        all_ok += ok
        all_ng += ng
        print("{} ok: {}, ng: {}".format(sheet, ok, ng))
    print("all ok: {}, all ng: {}".format(all_ok, all_ng))



def check_one_rgb_one_lab():
    all_train_rgb = json.load(open(r'./1118_train_rgb.json', 'r'))
    all_lab = json.load(open(r'./1118_lab.json', 'r'))
    lab_rgb = dict()
    for k, v in all_train_rgb.items():
        lab = ''.join(str(a) for a in all_lab[k])
        lab_rgb[lab] = ''.join(str(a) for a in v)
    vs = []
    print("diff labs: {}".format(len(lab_rgb)))
    for k, v in lab_rgb.items():
        if v not in vs:
            vs.append(v)
    print("diff rgbs: {}".format(len(list(set(vs)))))


def pass_unpass_show_each_oven_rgb(k2s, k1s, all_pass_no_pass):
    def gamma(a):
        if a > 0.04045:
            a = np.power((a + 0.055) / 1.055, 2.4)
        else:
            a /= 12.92

        return a

    inds = [i for i in range(1, 17)]
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'yellow', 'red', 'darkslateblue', 'turquoise',
              'blue', 'yellow', 'black', 'pink', 'red', 'green', 'cornflowerblue', 'purple', 'turquoise']
    # colors = ['pink'] * 15 + ['black', 'red']
    passes = []
    pass_, no_pass = 0, 0
    for ind in inds:
        # if ind in [7, 8]:
        js = json.load(open(r'./1122_rgb_js/dir_{}_rgb.json'.format(ind), 'r'))
        ks = list(js.keys())
        ks = [k for k in ks if k in k2s]
        if len(ks) > 0:
            print(ks, ind)
            for ii, k in enumerate(ks):
                if ii == 0:
                    plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind], label=ind)
                else:
                    plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in js[k]], color=colors[ind])
                passes.append(all_pass_no_pass[k])
                if len(all_pass_no_pass[k]) == 2:
                    pass_ += 1
                else:
                    no_pass += 1
    plt.grid()
    plt.legend()
    plt.show()
    print("pass: {}, no_pass: {}".format(pass_, no_pass))

    # k2通过, k1不通过
    # k2s = open(r'./k2.txt', 'r').readlines()[0].split(',')[:-1]
    # all_train_rgb = json.load(open(r'./1118_train_rgb.json', 'r'))
    # all_pass_no_pass = json.load(open(r'./1118_pass_no_pass.json', 'r'))
    # trains = list(all_train_rgb.keys())
    # k1s = [k for k in trains if k not in k2s]
    # show_each_oven_rgb(k2s, k1s, all_pass_no_pass)

    # ff2 = open(r'./k2.txt', 'w')
    # for k, v in all_pass_no_pass.items():
    #     if len(v) == 2:
    #         ff2.write(k + ',')


if __name__ == '__main__':


    # get rgb
    # main()

    # get lab
    # get_lab()
    get_lab1('8')

    # show_dir_ng_ok()

    # col1 = json.load(open(r'./dir_5_rgb_me.json', 'r'))
    # col2 = json.load(open(r'dir_5_rgb.json', 'r'))
    # for k, v in col1.items():
    #     print("myg: {}, me: {}, diff: {}".format(col2[k], col1[k], [col2[k][i] - col1[k][i] for i in range(3)]))





