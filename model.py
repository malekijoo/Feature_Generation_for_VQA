
import numpy as np
import tensorflow as tf

class FExt_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense2 = tf.keras.layers.Conv2D(64)
        self.dense1 = tf.keras.layers.AveragePooling2D()
        self.dropout = tf.keras.layers.Dense(2048)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)



print('model = FExt_Model()')