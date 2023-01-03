import os
import re
import glob
import yaml
import argparse
from pathlib import Path

class PathCfg:

    def __init__(self, dirname=None):

        self.root_path = './'
        self.yolo_path = './yolov7'
        self.cfgs_path = './cfgs'
        self.result_dir = './exp'
        self.coco_path = './coco'
        self.pred_filename = 'predictions.csv'
        self.exp_dir = (dirname if dirname else 'run')

        if not os.path.isdir('./coco'):
            Path(self.coco_path).mkdir(parents=True, exist_ok=True)
            # print('coco dataset directory is built: ', self.coco_path)

        if os.path.isdir('/content/gdrive/MyDrive/'):
            self.gdrive_path = '/content/gdrive/MyDrive'
            self.result_dir = '/content/gdrive/MyDrive/exp'
            self.yolo_path = '/content/gdrive/MyDrive/results/predn'

            self.gd_status = True
        else:
            self.gd_status = False

        self.save_path = Path(self.path_maker(self.result_dir + '/' + self.exp_dir, exist_ok=True)).resolve() # increment run
        self.save_path.mkdir(parents=True, exist_ok=True)  # make dir


    @staticmethod
    # written by https://github.com/WongKinYiu  yolov7/utils/general.py  increment_path fn
    def path_maker(path, exist_ok=True, sep='_'):
        # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
        path = Path(path)  # os-agnostic
        if (not path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            return f"{path}{sep}{n}"  # update path



if __name__ == '__main__':
    a = PathCfg()
    # print(a.path_maker('../data/')) # here we should use the parent directory address, in main it is ok to use

