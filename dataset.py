import yaml
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


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
            print('Downloading the dataset...')
            ds, ds_info = tfds.load(name="coco/2017",
                                    split=self.task,
                                    data_dir='./coco',
                                    shuffle_files=True,
                                    batch_size=self.params.batch,
                                    with_info=True,
                                    )
            # train_dataset, test_dataset = datasets["train"], datasets["test"] # if NOT split="train"

        except:
            print('There was an error downloding the dataset with `tfds`. \n'
                  'The dataset is downloaded from its source')
            subprocess.call('./scripts/get_coco.sh')
            coco_builder = tfds.builder("coco/2017")
            ds_info = coco_builder.info
            coco_builder.download_and_prepare(data_dir='./coco/')
            datasets = coco_builder.as_dataset(shuffle_files=True, batch_size=self.params.batch)
            ds = datasets[self.task]
            assert isinstance(ds, tf.data.Dataset)
            # ds_train = ds_train.repeat().shuffle(1024).batch(128)
            # ds_train = ds_train.prefetch(2)
            # features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
            # image, label = features['image'], features['label']

        self.ds = ds
        self.ds_info = ds_info
        self.hyp = self.ds_hyp()

    def ds_hyp(self):

        with open(self.params.data) as f:
            ds_hyp = yaml.load(f, Loader=yaml.SafeLoader)

        return ds_hyp


if __name__ == '__main__':
    coco = CoCo()
    ds_train, ds_info = coco.ds, coco.ds_info