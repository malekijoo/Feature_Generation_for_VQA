import os
import sh
import yaml
import argparse
import functools
import subprocess
import utils as ut
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
        self.download()

        # if cfgs.preprocessing:
            # self.ds = self.ds.map(functools.partial(preprocess, bgr=True),
            #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.hyp = cfgs.hyp
        self.hyp['names_no'] = [str(i) for i in range(self.hyp['nc'])]




        # size = (200, 200)
        # ds = ds.map(lambda img: smart_resize(img, size))
    def download(self):

        # try:
        # self = cls(cfgs)
        # print(f'Downloading the {self.task} dataset...')
        # ds, ds_inf = tfds.load(name="coco/2017",
        #                        split=self.task,
        #                        data_dir=self.ds_path,
        #                        with_info=True,
        #                        )
        # assert isinstance(ds, tf.data.Dataset), "DataSetError: ds is not an instance of tf.data.Dataset"
        # return ds, ds_inf
        if ut.check_dataset(cfgs):
            print(f'Tha dataset has already downloaded and unzipped in {self.path}!')

        else:
            print(f'\n the dataset is started to download ... \n the deafault pthe is %  {self.path}')
            subprocess.call('./scripts/get_coco.sh')



        # with open(file=)
        # coco_builder = tfds.builder("coco/2017", data_dir=self.ds_path)
        # ds_inf = coco_builder.info
        # coco_builder.download_and_prepare(download_dir=self.ds_path)
        # datasets = coco_builder.as_dataset()
        # ds = datasets[self.task]
        # shuffle_files = True, batch_size = self.params.batch
        # ds_train = ds_train.repeat().shuffle(1024).batch(128)
        # ds_train = ds_train.prefetch(2)
        # features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
        # image, label = features['image'], features['label']

    @classmethod
    def resize(cls, tg_size):
        print('amir ', tg_size)









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
