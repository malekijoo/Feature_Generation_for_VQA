# This is a sample Python script.
import yaml
import pandas
import argparse
from cfgs.base_cfg import Cfgs
import tqdm
import numpy as np
from model import FExt_Model
from dataset import CoCo
from cfgs.base_cfg import Cfgs
from yolo import YoloPred
import tensorflow as tf


def train(params):

    cfgs = Cfgs(params)
    print(cfgs)
    yolo = YoloPred(cfgs)
    # tg, df = yolo.img_extract('000000003694.jpg', top_k=False, conf_tr=0.3)
    coco = CoCo(cfgs=cfgs)
    # {'images', 'images_info', 'bbox', 'labels', 'num_boxes', 'weights'}
    dataloader = coco.dataloader
    model = FExt_Model()


    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        if batch_i == 1:
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = img.shape

            print(type(img), img.shape)
            print(type(targets), targets.shape)
            print(type(paths), paths)
            print(shapes)
            tg, df = yolo.img_extract('000000003694.jpg', top_k=False, conf_tr=0.3)
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')

    params = parser.parse_args()
    train(params)
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
