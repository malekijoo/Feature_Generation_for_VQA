import sh
import yaml
import argparse
import subprocess
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class CoCo:

    def __init__(self, params, task='train'):
        """
        dataset = {'images': A tensor of float32 and shape[1, height, widht, 3],
                   'images_info': A tensor of float32 and shape[1, 2],
                   'bbox': A tensor of float32 and shape[1, num_boxes, 4],
                   'labels': A tensor of int32 and shape[1, num_boxes],
                   'num_boxes': A tensor of int32 and shape[1, 1],
                   'weights': A tensor of float32 and shape[1, num_boxes]
                    }
        """
        self.task = task
        self.params = params

        try:
            if 'G' not in sh.du('-hs', Path('./coco')):
                print('Downloading the dataset...')
                ds, ds_inf = tfds.load(name="coco/2017",
                                       split=self.task,
                                       data_dir='./coco',
                                       shuffle_files=True,
                                       batch_size=self.params.batch,
                                       with_info=True,
                                       )
                # train_dataset, test_dataset = datasets["train"], datasets["test"] # if NOT split="train"
            else:
                coco_builder = tfds.builder("coco/2017", data_dir='./coco/')
                ds_inf = coco_builder.info
                coco_builder.download_and_prepare(download_dir='./coco/')
                datasets = coco_builder.as_dataset(shuffle_files=True, batch_size=self.params.batch)
                ds = datasets[self.task]
                assert isinstance(ds, tf.data.Dataset)
        except:
            print('There was an error downloding the dataset with `tfds`. \n'
                  'The dataset is downloaded from its source')
            if 'G' not in sh.du('-hs', Path('./coco')):
                subprocess.call('./scripts/get_coco.sh')
            coco_builder = tfds.builder("coco/2017", data_dir='./coco/')
            ds_inf = coco_builder.info
            coco_builder.download_and_prepare(download_dir='./coco/')
            datasets = coco_builder.as_dataset(shuffle_files=True, batch_size=self.params.batch)
            ds = datasets[self.task]
            assert isinstance(ds, tf.data.Dataset)
            # ds_train = ds_train.repeat().shuffle(1024).batch(128)
            # ds_train = ds_train.prefetch(2)
            # features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
            # image, label = features['image'], features['label']

        self.ds = ds
        self.ds_info = ds_inf
        self.hyp = self.ds_hyp()

    def ds_hyp(self):

        with open(self.params.data) as f:
            ds_hyp = yaml.load(f, Loader=yaml.SafeLoader)

        return ds_hyp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--batch', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch', type=int, default=300, help='input the number of epochs')
    pr = parser.parse_args()

    coco = CoCo(pr)
    ds_train, ds_info = coco.ds, coco.ds_info
    coco_hyp = coco.hyp
