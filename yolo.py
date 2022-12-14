import numpy as np
import pandas as pd
import argparse
from cfgs.base_cfg import Cfgs

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class YoloPred:

    def __init__(self, cfgs):

        self.yolo_path = cfgs.yolo_path
        self.pred_name = cfgs.pred_filename

    @property
    def pred_df(self):

        _pred_df = pd.read_csv(self.yolo_path + '/' + self.pred_name)
        _pred_df.columns = ['x1', 'y1', 'x2', 'y2', 'conf', 'cls', 'path']
        _pred_df = _pred_df.drop_duplicates()

        _pred_df['conf'] = _pred_df['conf'].astype(np.float64)

        _pred_df = _pred_df.astype({'x1': np.float64,
                                    'y1': np.float64,
                                    'x2': np.float64,
                                    'y2': np.float64,
                                    })

        _pred_df['x1'] = _pred_df['x1'].apply(np.floor).astype(np.int32)
        _pred_df['y1'] = _pred_df['y1'].apply(np.floor).astype(np.int32)
        _pred_df['x2'] = _pred_df['x2'].apply(np.ceil).astype(np.int32)
        _pred_df['y2'] = _pred_df['y2'].apply(np.ceil).astype(np.int32)

        # _pred_df[_pred_df.columns[0:4]] = YoloPred.xyxy2xywh(_pred_df[_pred_df.columns[0:4]].to_numpy())
        return _pred_df

    @pred_df.setter
    def pred_df(self, predn):
        self._pred_df = predn

    def img_extract(self, key, conf_tr=0, top_k=False, k=100):
        """
        :param key: the image filename, e.x. '000000003694.jpg'
        :param top_k: Boolean type, select the type of picking
                      the Bounding Box (BB). e.x. the top-k with the highest confidence,
                      or the number of BB with higher a specific confidence
        :param conf_tr: the threshold which apply to confidence
        :param k: the top k number of BB with the highest confidence.
        :return: Numpy array of BBox, a dataframe of BB with 7 columns, the name of the img

        """
        if conf_tr >= 1:
            conf_tr /= 100
            # print(f'Confidence threshold is tr={conf_tr}%. It means the BBox lower than the tr will be filtered')

        spilited_key = key.split('/')
        key = [x for x in spilited_key if '.jpg' in x][0]
        img_name = 'coco/images/train2017/' + key
        # if self.pred_df[self.pred_df['path'] == img_name].size == 0:
        #     return [False, key]
        dummy_df = self.pred_df[self.pred_df['path'] == img_name].sort_values(by=['conf'], ascending=False)

        if conf_tr > 0:
            # print(f'Confidence threshold is tr={conf_tr}. It means the BBox lower than the tr will be filtered')
            dummy_df = dummy_df[dummy_df['conf'] >= conf_tr]

        if top_k:
            dummy_df = dummy_df.head(k)

        tg = dummy_df[['x1', 'y1', 'x2', 'y2']].values.tolist()

        return np.array(tg), dummy_df, key



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./data/coco.yaml', help='hyper parameter of dataset ')
    parser.add_argument('-i', '--info', type=str, default='', help='information of  ')
    parser.add_argument('-t', '--task', type=str, default='train', help='train or test')
    parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='input the number of epochs')
    parser.add_argument('-p', '--preprocessing', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--exp-dir', type=str, default='run', help='directory of result')

    params = parser.parse_args()
    cfgs = Cfgs(params)

    yolo = YoloPred(cfgs)
    # print(yolo.pred_df.head())
    k = 'coco/images/train2017/000000148246.jpg'
    # print(yolo.img_extract(k, conf_tr=30))




