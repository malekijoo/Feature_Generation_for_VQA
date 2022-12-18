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
        in Tensorflow : https://www.tensorflow.org/datasets/catalog/coco
                FeaturesDict({
                'image': Image(shape=(None, None, 3), dtype=uint8),
                'image/filename': Text(shape=(), dtype=string),
                'image/id': int64,
                'objects': Sequence({
                'area': int64,
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
                'id': int64,
                'is_crowd': bool,
                'label': ClassLabel(shape=(), dtype=int64, num_classes=80),
                }),
                })
        """

        self.hyp = cfgs.hyp
        self.path = cfgs.coco_path
        self.preprocessing = cfgs.preprocessing

        print(self.path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.task = cfgs.task
        self.ds, self.ds_info = self._download()

        if self.preprocessing:

            self.ds = self.ds.map(functools.partial(preprocess, bgr=True),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # resize to 640*640
            self.ds = self.ds.map(lambda img: tf.image.resize_with_pad(img, target_height=640, target_width=640))

    def _download(self):

        print(f'Downloading the {self.task} dataset...')
        _ds, _ds_inf = tfds.load(name="coco/2017",
                               split=self.task,
                               data_dir=self.path,
                               with_info=True,
                               )
        assert isinstance(_ds, tf.data.Dataset), "DataSetError: ds is not an instance of tf.data.Dataset"
        return _ds, _ds_inf

        # if ut.check_dataset(cfgs):
        #     print(f'Tha dataset has already downloaded and unzipped in {self.path}!')
        #
        # else:
        #     print(f'\n the dataset is started to download ... \n the deafault pthe is %  {self.path}')
        #     subprocess.call('./scripts/get_coco.sh')
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

    @staticmethod
    def bb_crop_image(img, tg, tg_size=(224, 224)):
        # Batch, x, y, w, h = tg # target should be 5D array,
        height, width = tg_size
        img = [tf.image.crop_to_bounding_box(img, x[0], x[1], x[2], x[3]) for x in tg]
        cropped_img = img.map(lambda img: tf.image.resize_with_pad(img, target_height=height, target_width=width))

        return cropped_img




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=1, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')
    pr = parser.parse_args()

    cfgs = Cfgs(pr)
    coco = CoCo(cfgs=cfgs)
    # ds_train, ds_info = coco.ds, coco.ds_info
    # coco_hyp = coco.hyp
