import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input


class FExt:
    def __init__(self):
        self.base_model = Xception(weights='imagenet', include_top=True)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def __call__(self, x, preprocessing=True):
        if preprocessing:
            x = preprocess_input(x)
        return self.model.predict(x)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(**kwargs)
    #     self.base_model = Xception(weights='imagenet', include_top=True)
    #     # avg pool (GlobalAveragePooling  (None, 2048)
    #     # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    #
    # def call(self, inputs, training=False):
    #     x = preprocess_input(inputs)
    #     x = self.base_model(x)
    #     return self.base_model.get_layer('avg_pool').output


if __name__ == '__main__':
    print('model = FExt()')
    FExt()
