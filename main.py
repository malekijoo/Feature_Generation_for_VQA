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


def train(params):

    cfgs = Cfgs(params)
    yolo = YoloPred(cfgs)
    # tg, df = yolo.img_extract('000000003694.jpg', top_k=False, conf_tr=0.3)
    coco = CoCo(cfgs=cfgs)
    # {'images', 'images_info', 'bbox', 'labels', 'num_boxes', 'weights'}
    dataloader = coco.dataloader
    model = FExt()
    print(type(dataloader))
    vqa_dict = {}
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        if batch_i == 1:
            img = img.numpy()
            targets = targets.numpy()
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = img.shape

            filename = paths[0]
            print('\n ** \n filename ', filename, img.shape)
            filename = '000000018908.jpg'
            tg, df, key = yolo.img_extract(filename, top_k=False, conf_tr=0.3)
            print('tg shape ', np.array(tg).shape)
            print('key ', key)
            bb_imgs = coco.bb_crop_image(img, tg)
            print('\n ********** \n ')

            print('bb_imgs ', type(bb_imgs), bb_imgs.shape)
            print('\n ********** \n ')
            print(type(img), img.shape)
            print(type(targets), targets.shape)
            print(type(paths), paths)
            print(shapes)

            filename = f'COCO_val2014_{key}.npz'
            np.savez(filename, )
            output = model(bb_imgs)

            vqa_dict['x'] = output
            vqa_dict['image_w'] = width
            vqa_dict['image_h'] = height
            vqa_dict['num_bbox'] = len(tg)
            vqa_dict['bbox'] = tg
            print('output shape ', output.shape)

            npz_dict = npz_dict.copy()
            np.savez(str(Path(cfgs.save_path, filename)), npz_dict)

            # coco.bb_crop_image(im, tg)
            # xx = tf.data.frosilice().batch(32)
            # output = model(xx)
            # output + path will be save like mcan-vqa



    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    #
    # block4_pool_features = model.predict(x)

    # cfgs = Cfgs(params)
    # print(cfgs.gdrive)
    # coco = CoCo(params)
    # ds_train, ds_info = coco.ds, coco.ds_info
    # coco_hyp = coco.hyp
    # print('coco ', coco)
    # print(type(ds_train), ds_train)
    # print(type(ds_info), ds_info)
    # print(coco_hyp['names_no'])
    # print('coco hype ', coco_hyp)

#
# image type  <class 'torch.Tensor'> torch.Size([1, 3, 128, 672])
# target  <class 'torch.Tensor'> torch.Size([2, 6])
# paths  <class 'tuple'> ('coco/images/train2017/000000207005.jpg',)
# shapes (((104, 640), ((1.0, 1.0), (16.0, 12.0))),)
# Confidence threshold is tr=0.3. It means the BBox lower than the tr will be filtered
#   0% 1/118287 [00:00<4:15:52,  7.70it/s]
#  image type  <class 'torch.Tensor'> torch.Size([1, 3, 128, 672])
# target  <class 'torch.Tensor'> torch.Size([4, 6])
# paths  <class 'tuple'> ('coco/images/train2017/000000029005.jpg',)
# shapes (((109, 640), ((1.0, 1.0), (16.0, 9.5))),)
# Confidence threshold is tr=0.3. It means the BBox lower than the tr will be filtered
#   0% 2/118287 [00:00<4:20:12,  7.58it/s]
#  image type  <class 'torch.Tensor'> torch.Size([1, 3, 128, 672])
# target  <class 'torch.Tensor'> torch.Size([11, 6])
# paths  <class 'tuple'> ('coco/images/train2017/000000118638.jpg',)
# shapes (((111, 640), ((1.0, 1.0), (16.0, 8.5))),)
# Confidence threshold is tr=0.3. It means the BBox lower than the tr will be filtered
#   0% 3/118287 [00:00<4:18:42,  7.62it/s]
#  image type  <class 'torch.Tensor'> torch.Size([1, 3, 160, 672])
# target  <class 'torch.Tensor'> torch.Size([12, 6])
# paths  <class 'tuple'> ('coco/images/train2017/000000268838.jpg',)
# shapes (((128, 640), ((1.0, 1.0), (16.0, 16.0))),)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=1, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')

    params = parser.parse_args()
    train(params)
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# ['val2014/COCO_val2014_000000306395.jpg.npz']
#
# the
# loop
# 0 is starting
# val2014 / COCO_val2014_000000306395.jpg.npz
#
# 306395
# (2048, 25)
# ['x', 'image_w', 'bbox', 'num_bbox', 'image_h']
# bbox
# shape = (25, 4)
# image_w = 640
# num_bbox = 25
# image_h = 480
# (25, 2048)
# Pre - Loading: [0 | 40504]