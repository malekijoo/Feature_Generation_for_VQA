import os
import sh
import yaml
import argparse
import functools
import subprocess
import numpy as np
from cfgs.base_cfg import Cfgs
import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from kerod.core.standard_fields import BoxField
from kerod.dataset.preprocessing import preprocess, expand_dims_for_single_batch


class CoCo:

    def __init__(self, cfgs):
        """
        dataset = {'images': A tensor of float32 and shape[1, height, widht, 3],
                   'images_info': A tensor of float32 and shape[1, 2],
                   'bbox': A tensor of float32 and shape[1, num_boxes, 4],
                   'labels': A tensor of int32 and shape[1, num_boxes],
                   'num_boxes': A tensor of int32 and shape[1, 1],
                   'weights': A tensor of float32 and shape[1, num_boxes]
                    }
        """
        self.path = cfgs.coco_path
        print(self.path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.task = cfgs.task
        # self.ds, self.ds_info = self.download()


        if cfgs.preprocessing:
            self.ds = self.ds.map(functools.partial(preprocess, bgr=True),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.hyp = cfgs.hyp
            self.hyp['names_no'] = [str(i) for i in range(self.hyp['nc'])]

        # size = (200, 200)
        # ds = ds.map(lambda img: smart_resize(img, size))
    @classmethod
    def download(cls, cfgs):

        # try:
        self = cls(cfgs)
        print(f'Downloading the {self.task} dataset...')
        ds, ds_inf = tfds.load(name="coco/2017",
                               split=self.task,
                               data_dir=self.ds_path,
                               with_info=True,
                               )
        assert isinstance(ds, tf.data.Dataset)
        return ds, ds_inf






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
    coco = CoCo(cfgs=cfgs)
    # ds_train, ds_info = coco.ds, coco.ds_info
    # coco_hyp = coco.hyp
