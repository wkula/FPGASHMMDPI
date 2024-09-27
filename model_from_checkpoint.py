import tensorflow as tf
from read_data import BSData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os
import logging
from keras.callbacks import CSVLogger


def CNN_model(shape=(9, 8)):
    mult = 1
    for i in shape:
        mult *= i
    din = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(128, 3, activation='linear')(din)
    x = tf.keras.layers.Conv2D(28, 3, activation='linear')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='linear')(x)

    y = tf.keras.layers.Flatten()(din)
    y = tf.keras.layers.Dense(1024, activation='linear')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(1024, activation='linear')(y)
    y = tf.keras.layers.Dropout(0.2)(y)

    z = tf.keras.layers.Dense(1024, activation='relu')(y)
    z = tf.keras.layers.Dropout(0.2)(z)
    z = tf.keras.layers.Dense(1024, activation='relu')(z)
    z = tf.keras.layers.Dropout(0.2)(z)

    y = tf.keras.layers.concatenate([y, z])
    y = tf.keras.layers.Dense(256, activation='linear')(y)

    x = tf.keras.layers.concatenate([x, y])

    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation='linear')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(din, x)
    return model


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("Creating model")
    model = CNN_model(shape=(250, 8, 1))
    model.compile(optimizer='Adam',
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy', F1Score(num_classes=1)])

    checkpoint_filepath = './tmpCNN/checkpoint'

    model.load_weights(checkpoint_filepath)
    model.save('CNN_2D_binLR_loaded.h5')
