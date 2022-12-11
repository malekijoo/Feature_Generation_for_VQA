# This is a sample Python script.
import yaml
import pandas
import argparse
from cfgs import path_cfg as pcfg

import numpy as np
from dataset import CoCo
from cfgs.base_cfg import Cfgs
import tensorflow as tf


def train(params):
    a = pcfg.PathCfg()
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
    parser.add_argument('-i', '--info', type=str, default='./data/coco.yaml', help='Information ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    pr = parser.parse_args()
    train(pr)
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
