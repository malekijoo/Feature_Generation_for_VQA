
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

class FExt(tf.keras.Model):
    """
    اول باید تمام باندینگ باکس ها بیاد تو resnet101
    بعد اون قسمت ابجکتش در فیچر مپ اخر جدا بشه
    و تمامشون ذخیره بشه.
    چجوری ذخیره بشه رو میتونی از کد مرجع اون دوتا فایلش رو بخونی
    """
    def __init__(self):
        super().__init__()

        self.base_model = Xception(weights='imagenet', include_top=False)
        # avg pool (GlobalAveragePooling  (None, 2048)
        # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def call(self, inputs, training=False):
        x = preprocess_input(inputs)
        _ = self.base_model(x)
        return self.base_model.get_layer('avg_pool').output


if __name__ == '__main__':
    print('model = FExt()')
    FExt()