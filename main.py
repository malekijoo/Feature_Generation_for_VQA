# This is a sample Python script.
import yaml
import pandas
import argparse

import numpy as np
import dataset as dt
import tensorflow as tf


def train(params):

    coco = dt.CoCo(params)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--batch', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch', type=int, default=300, help='input the number of epochs')
    pr = parser.parse_args()

    train(pr)
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
