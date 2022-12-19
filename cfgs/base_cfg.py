import os
import yaml
import argparse
import random
from cfgs.path_cfg import PathCfg
from types import MethodType
import pathlib
from pathlib import Path
import numpy as np

class Cfgs(PathCfg):

    def __init__(self, parser_args, mode='base'):

        super(Cfgs, self).__init__(parser_args.exp_dir)

        self.SEED = random.randint(0, 99999999)
        self.hyp = Cfgs.yaml_reader(parser_args.yaml)
        self.hyp['cls_no'] = [str(i) for i in range(self.hyp['nc'])]

        self.parser_args = Cfgs.parse_to_dict(parser_args)
        self.mode = mode
        self.epoch = parser_args.epoch
        self.batch_size = parser_args.batch
        self.preprocessing = parser_args.preprocessing
        self.task = parser_args.task
        self.yolo_pred_filename = 'predictions.csv'
        self.imgsize = (640, 640)

    @staticmethod
    def parse_to_dict(args):
      args_dict = {}
      for arg in dir(args):
        if not arg.startswith('_'): # and not isinstance(getattr(args, arg), MethodType):
          if getattr(args, arg) is not None:
            args_dict[arg] = getattr(args, arg)

      return args_dict
    @staticmethod
    def yaml_reader(path):
      with open(path) as f:
        _yaml = yaml.load(f, Loader=yaml.SafeLoader)
      return _yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='../data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')
    pr = parser.parse_args()

    a = Cfgs(pr)
    # print(a.__dict__)

