# coding=utf-8
import os
import cv2
import numpy as np

def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def get_roi_value(image, a, b, c, d, ind):

    mask = np.zeros((image.shape), dtype=np.uint8)
    pts = np.array([[a, b, c, d]], dtype=np.int32)
    cv2.fillPoly(mask, pts, (255,255,255))
    values = image[np.where((mask == (255,255,255)).all(axis=2))]
    cv2.rectangle(img, d, b, (0, 255, 255), 3)
    cv2.imwrite('./diamond_mask{}.png'.format(ind), img)

    return values



dir = r'C:\Users\15974\Desktop\蔡司-膜色1\1'
ims = os.listdir(dir)
ims = [a for a in ims if 'bmp' in a]
for ind, im in enumerate(ims):
    im_path = os.path.join(dir, im)
    img = imread(im_path)
    p1, p2, p3, p4 = [1289, 1000], [1289, 1040], [1249, 1040], [1249, 1000]
    roi_value = get_roi_value(img, p1, p2, p3, p4, ind).mean(axis=0)
    print(roi_value)







