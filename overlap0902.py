import cv2
import json
import numpy as np
'''
for data0812 blue imgs

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


def main(single_dir_col, dir_index, dir_path):
    import glob
    config_file = os.path.join(dir_path, r'config.json')
    with open(config_file) as f:
        conf = json.load(f)
        color_lower = tuple(conf.get("color_lower"))
        color_upper = tuple(conf.get("color_upper"))
        area_threshold = conf.get("area_threshold")
        color_thresholds = tuple(conf.get("color_thresholds"))

    print("color_lower:", color_lower)
    print("color_upper:", color_upper)
    im_paths = glob.glob(os.path.join(dir_path, r"*.bmp").format(dir_path))
    im_names = os.listdir(dir_path)
    for ind, im_path in enumerate(im_paths):
        im_name = int(im_path.split('\\')[-1][:-4])
        img = imread(im_path)
        color, string, sig, draw = pipeline(img, color_lower, color_upper, area_threshold, color_thresholds)
        pre_path = r'D:\work\project\zeiss_poc\{}'.format(dir_path.split('\\')[-1])
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        if not cv2.imwrite(os.path.join(pre_path, im_names[ind][:-4]+"_overlap.bmp"), img):
            print('img not save ..')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        single_dir_col["{}_{}".format(dir_index + 20, im_name)] = [str(a) for a in color]


if __name__ == '__main__':
    import os
    import json
    dir_n = 8
    base_dir = r"D:\work\project\卡尔蔡司AR镀膜\poc\0812\0812"
    all_col3 = dict()
    for i in range(1, dir_n+1):
        dir_path = os.path.join(base_dir, str(i))
        main(all_col3, i, dir_path)
    data = json.dumps(all_col3)
    with open('./all_col3_0821.json', 'w') as js_file:
        js_file.write(data)
