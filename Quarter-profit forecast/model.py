from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import xlrd
from keras.callbacks import ModelCheckpoint
import os
from jqdatasdk import *

def model():
    # LSTM 神经网络搭建
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(70, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, dropout=0.1, return_sequences=False))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mse')
    return model