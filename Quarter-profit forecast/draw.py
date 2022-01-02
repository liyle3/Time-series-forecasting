import matplotlib.pyplot as plt
from pylab import *

def draw(model, x_predict, y_predict, y_actual):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    y_predict = model.predict(x_predict)*1e6
    x = [i for i in range(1, 11)]
    plt.figure(figsize = (16, 9))
    plt.axis([0, 12, 5e6, 5e8])
    # plt.ylim([5e6, 5e8])
    plt.plot(x, y_actual, marker='o', ms=8, mec ='r', mfc = 'w', label = 'origin')
    plt.plot(x, y_predict[0], marker='*', ms = 8, label = 'prediction')
    plt.ylabel(f"利润", fontsize=12)
    plt.xlabel(f"季度", fontsize=12)
    plt.legend()
    plt.show()

    new_model = tf.keras.models.load_model('saved_model/my_model')