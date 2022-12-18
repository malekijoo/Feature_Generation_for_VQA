
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception  import Xception
from tensorflow.keras.applications.vgg19 import preprocess_input

class FExt_Model(tf.keras.Model):
    """
    اول باید عکس بیاد تو resnet101
    بعد اون قسمت ابجکتش در فیچر مپ اخر جدا بشه
    به یه conv2d mean pool و dense داده بشه
    """
    def __init__(self):
        super().__init__()

        base_model = Xception(weights='imagenet', include_top=False)
        # avg pool (GlobalAveragePooling  (None, 2048)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)




    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        # if training:
        #     x = self.dropout(x, training=training)
        # return self.dense2(x)


if __name__ == '__main__':
    print('model = FExt_Model()')
    FExt_Model()