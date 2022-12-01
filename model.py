
import numpy as np
import tensorflow as tf

class FExt_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
          x = self.dropout(x, training=training)
        return self.dense2(x)



print('model = FExt_Model()')