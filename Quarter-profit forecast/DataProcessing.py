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
import warnings
warnings.filterwarnings("ignore")

def NAN_Replace(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == '--':
                data[i, j] = 0

    return data

def DataProcessing(path):
    excel = pd.read_excel(path, engine='openpyxl')
    raw = np.array(excel)
    y_data = []
    x_data = []
    for arr in raw:
        x_data.append(arr[5:75])
        y_data.append(arr[75:])

    x_data = NAN_Replace(np.array(x_data))
    y_data = NAN_Replace(np.array(y_data))
    x_data = x_data.astype('float64')
    y_data = y_data.astype('float64')

    x_train = x_data[:350]
    y_train = y_data[:350]

    x_val = x_data[350:400]
    y_val = y_data[350:400]

    x_test = x_data[400:]
    y_test = y_data[400:]

    x_train = x_train.reshape((350, 70, 1))/1e6
    y_train = y_train.reshape((350, 10, 1))/1e6
    x_val = x_val.reshape((50, 70, 1))/1e6
    y_val = y_val.reshape((50, 10, 1))/1e6
    x_test = x_test.reshape((32, 70, 1))/1e6
    y_test = y_test.reshape((32, 10, 1))/1e6

    return x_data, y_data, x_train, y_train, x_val, y_val, x_test, y_test