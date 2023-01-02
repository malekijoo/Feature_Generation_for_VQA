# This is a sample Python script.
import yaml
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from model import FExt
from dataset import CoCo
from pathlib import Path
from yolo import YoloPred
from cfgs.base_cfg import Cfgs
import tensorflow as tf


def extractor(params):

    cfgs = Cfgs(params)
    yolo = YoloPred(cfgs)
    # tg, df = yolo.img_extract('000000003694.jpg', top_k=False, conf_tr=0.3)
    coco = CoCo(cfgs=cfgs)
    # {'images', 'images_info', 'bbox', 'labels', 'num_boxes', 'weights'}
    dataloader = coco.dataloader
    model = FExt()
    vqa_dict = {}

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        if batch_i == 1:
            img = img.numpy()
            targets = targets.numpy()
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = img.shape

            filename = paths[0]
            # filename = '000000018908.jpg'
            tg, df, key = yolo.img_extract(filename, top_k=False, conf_tr=0.3)
            bb_imgs = coco.bb_crop_image(img, tg)
            output = model(bb_imgs, preprocessing=cfgs.preprocessing)
            print('output   _____1  ', type(output), output.shape)

            vqa_dict['x'] = output
            vqa_dict['image_w'], vqa_dict['image_h'] = width, height
            vqa_dict['bbox'], vqa_dict['num_bbox'] = tg, output.shape[0]
            save_2_numpyz(cfgs.save_path, key, vqa_dict)




def save_2_numpyz(path, key, dic):
    filename = f'COCO_val2014_{key}.npz'
    npz_dict = dic.copy()
    np.savez(str(Path(path, filename)), npz_dict)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='extractor', help='extractor or test')
    parser.add_argument('-b', '--batch', type=int, default=1, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')

    params = parser.parse_args()
    extractor(params)
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")

