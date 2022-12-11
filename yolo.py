import numpy as np
import pandas as pd
from cfgs.base_cfg import Cfgs

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class YoloPred:

    def __init__(self, cfgs):

        self.yolo_path = cfgss
        self.pred_name = 'predictions.csv'

    @property
    def pred_df(self):

        _pred_df = pd.read_csv(cfgss + '/' + self.pred_name)
        _pred_df.columns = ['x1', 'y1', 'x2', 'y2', 'conf', 'cls', 'path']
        _pred_df = _pred_df.drop_duplicates()
        print('before drop duplications', _pred_df.shape)

        # print(_pred_df[_pred_df['x2'].str.contains('x2')])
        _pred_df['conf'] = _pred_df['conf'].astype(np.float64) # * 100, 3)
        _pred_df = _pred_df.astype({'x1': np.float64,
                                    'y1': np.float64,
                                    'x2': np.float64,
                                    'y2': np.float64,
                                    })

        _pred_df['x1'] = _pred_df['x1'].apply(np.floor).astype(np.int16)
        _pred_df['y1'] = _pred_df['y1'].apply(np.floor).astype(np.int16)
        _pred_df['x2'] = _pred_df['x2'].apply(np.ceil).astype(np.int16)
        _pred_df['y2'] = _pred_df['y2'].apply(np.ceil).astype(np.int16)

        return _pred_df


    @pred_df.setter
    def pred_df(self):
        self._pred_fn = _pred_name

    def pred_group_by_key(self):
        pass
        # return

    @staticmethod
    def img_extract(df, key):
        spilited_key = key.split('/')
        key = [x for x in spilited_key if '.jpg' in x][0]
        akey = 'coco/images/train2017/' + key
        extr = df[df['path'] == akey]
        print(extr)






if __name__ == '__main__':
    cfgss = './yolov7'
    yolo = YoloPred(cfgss)
    print("here in main ", yolo.pred_df.shape)
    k = 'coco/images/train2017/000000148246.jpg'
    yolo.img_extract(yolo.pred_df, k)




