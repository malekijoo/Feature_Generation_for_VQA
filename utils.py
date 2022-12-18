import os.path

import cv2
import argparse
import numpy as np
from cfgs.base_cfg import Cfgs
from pathlib import Path


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def check_dataset(cfgs):

    item = ['train', 'val', 'test']

    list_files = [value for key, value in cfgs.hyp.items() if key in item]
    a_exist = [True if os.path.isfile(f) else False for f in list_files]

    if np.all(a_exist):
        return True
    elif not np.any(a_exist):
        return False
    elif np.any(a_exist):
        raise ValueError('Check dataset files. some of the .txts are not available')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')

    pr = parser.parse_args()
    cfgs = Cfgs(pr)

    ds_status = check_dataset(cfgs)
    print(ds_status)