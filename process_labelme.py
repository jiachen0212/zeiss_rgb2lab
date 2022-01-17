# coding=utf-8
import argparse
import base64
import json
import os
import os.path as osp
import warnings
import cv2
from shutil import copyfile
import PIL.Image
import yaml

from labelme import utils


def main(json_file):

    alist = os.listdir(json_file)
    alist = [a for a in alist if '.json' in a]

    for i in range(0, len(alist)):
        base_im_name = alist[i]
        path = os.path.join(json_file, alist[i])
        data = json.load(open(path, 'r'))

        out_dir = osp.basename(path).replace('.', '_')
        out_dir = osp.join(osp.dirname(path), out_dir)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.png'.format(base_im_name[:-5])))
        utils.lblsave(osp.join(out_dir, path[0:len(path) - 4] + 'png'), lbl)

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)



def post_process(dir_):
    save_image_dir = r'C:\Users\15974\Desktop\image'
    save_label_dir = r'C:\Users\15974\Desktop\label'
    files = os.listdir(dir_)
    imgs = [a for a in files if ".bmp" in a]
    labels = [a for a in files if "png" in a]
    assert len(imgs) == len(labels)
    print("data size: {}".format(len(labels)))
    for ind, img in enumerate(imgs):
        img_path = os.path.join(dir_, img)
        label_path = os.path.join(dir_, labels[ind])
        new_path1 = os.path.join(save_image_dir, img)
        copyfile(img_path, new_path1)
        new_path2 = os.path.join(save_label_dir, labels[ind])
        copyfile(label_path, new_path2)


if __name__ == '__main__':

    json_file = r'C:\Users\15974\Desktop\zeiss_膜色缺陷_所有数据\zeiss_膜色缺陷2'
    # main(json_file)

    # post_process
    post_process(json_file)
