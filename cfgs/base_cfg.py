import argparse
import os, random
from types import MethodType
import pathlib
from pathlib import Path
import numpy as np

class Cfgs:

    def __init__(self, parser_args, mode='base'):
        self.mode = mode
        self.parser_args = self.parse_to_dict(parser_args)
        self.SEED = random.randint(0, 99999999)
        self.path = {}

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_'): # and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--task', type=str, default='train', help='train or test')
    parser.add_argument('--batch', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch', type=int, default=300, help='input the number of epochs')
    pr = parser.parse_args()
    a = Cfgs(pr)
