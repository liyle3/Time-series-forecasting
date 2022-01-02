from keras import layers
from keras import models
import numpy as np
import pandas as pd
import os
from DataProcessing import *
from model import *

def main():
    path = "./DataSet/机械设备.xlsx"
    x_data, y_data, x_train, y_train, x_val, y_val, x_test, y_test = DataProcessing(path)

    model = model()
    model.fit(x_train, y_train, epochs=20, batch_size= 128, validation_data=(x_val,y_val))
    model.evaluate(x_test, y_test)
    model.save('saved_model/my_model')
    x_predict = x_data[402] #402
    y_actual = y_data[402]

    x_predict = x_predict.reshape((1, 70, 1))/1e6

    draw(mode, x_predict, y_predict, y_actual)


if __name__==__main__:
    main()