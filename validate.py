import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Input, BatchNormalization
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




if __name__ == "__main__":

    data = pd.read_csv('data_set_csv.csv', sep=',')

    # calculate the index to split the data into training and validation sets
    # *[rozdělení dat na trénovací a validační 80% trénovací + 20% validační]
    l = round(len(data)*0.8)
    # split the data into training and validation sets
    train = data.iloc[:l, :]
    val = data.iloc[l:, :]

    # extract the input and output values for the training set
    # [vstupem do neuronky jsou nejnižší dvě rezonanční frekvence]
    x_train = train.iloc[:, 0:2].values
    # [výstupní hodnoty pro trénovací možinu je vektor 100 hodnot 0-1 popisující design]
    y_train = train.iloc[:, 3:].values

    # extract the input and output values for the validation set
    x_val = val.iloc[:, 0:2].values
    y_val = val.iloc[:, 3:].values

    run = 'sweep_23'
    model = keras.models.load_model(f'training/{run}/best_model.h5')
    # RMSE = keras.metrics.RootMeanSquaredError()
    RMSE = keras.metrics.BinaryAccuracy()

    pred = model.predict(x_val)

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(pred[0, :], label='predictions')
    # ax.plot(y_val[0, :], label='targets')
    sns.heatmap(y_val - pred)
    # ax.set_title(f'Title')
    # ax.set_ylabel('deviation um')
    # ax.set_xlabel('samples over time (chronological)')
    # ax.legend(loc='lower right')
    # ax.grid('major')
    plt.show()


