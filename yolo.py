import numpy as np
import pandas as pd
from cfgs.base_cfg import Cfgs

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class YoloPred:

    def __init__(self, cfgs):

        self.yolo_path = cfgs.yolo_path
        self.pred_name = cfgs.yolo_pred_filename

    @property
    def pred_df(self):

        _pred_df = pd.read_csv(self.yolo_path + '/' + self.pred_name, index_col=[0])
        _pred_df.columns = ['x1', 'y1', 'x2', 'y2', 'conf', 'cls', 'path']
        _pred_df = _pred_df.drop_duplicates()

        _pred_df['conf'] = _pred_df['conf'].astype(np.float64)
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

    def img_extract(self, key, conf_tr=0, top_k=False, k=10):
        """
        :param key: the image filename, e.x. '000000003694.jpg'
        :param top_k: Boolean type, select the type of picking
                      the Bounding Box (BB). e.x. the top-k with the highest confidence,
                      or the number of BB with higher a specific confidence
        :param conf_tr: the threshold which apply to confidence
        :param k: the top k number of BB with the highest confidence.
        :return: a dataframe of BB with 7 columns
        """
        if conf_tr >= 1:
            conf_tr /= 100
        print(conf_tr)

        spilited_key = key.split('/')
        key = [x for x in spilited_key if '.jpg' in x][0]
        img_name = 'coco/images/train2017/' + key
        dummy_df = self.pred_df[self.pred_df['path'] == img_name].sort_values(by=['conf'], ascending=False)
        print(dummy_df.shape)
        if top_k:
            dummy_df = dummy_df.head(k)
        if conf_tr > 0:
            dummy_df = dummy_df[dummy_df['conf'] >= conf_tr]

        return dummy_df




if __name__ == '__main__':
    cfgss = './yolov7'
    yolo = YoloPred(cfgss)
    k = 'coco/images/train2017/000000148246.jpg'
    yolo.img_extract(yolo.pred_df, k)




