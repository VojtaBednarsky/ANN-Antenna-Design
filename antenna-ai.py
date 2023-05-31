from tensorflow import keras
from keras.layers import Dense, Dropout, Input
import tensorflow as tf
from keras import regularizers
import pandas as pd
import os

# zobrazeni vah a nastaveni
#https://netron.app/

data = pd.read_csv('data_set_csv.csv', sep=';')

l = round(len(data)*0.8)
train = data.iloc[:l, :]
val = data.iloc[l:, :]

x_train = train.iloc[:, 0:2].values
y_train = train.iloc[:, 3:-1].values

x_val = val.iloc[:, 0:2].values
y_val = val.iloc[:, 3:-1].values


initializer = keras.initializers.RandomUniform()
rmse = 99999
x = [x for x in range(100,600,100)]
for i, l  in enumerate(x) :
    
    model = keras.Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=l, activation='relu', kernel_initializer=initializer))
    model.add(Dense(units=200, activation='relu', kernel_initializer=initializer))    
              # ,
              #       kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
              #       bias_regularizer=regularizers.L2(1e-4),
              #       activity_regularizer=regularizers.L2(1e-5)))
    model.add(Dropout(.10))
    model.add(Dense(units=100, activation='sigmoid', kernel_initializer=initializer))
    
    lr = 0.01
    opt = keras.optimizers.Adam(learning_rate=lr)
    m = keras.metrics.RootMeanSquaredError()
    loss = keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=opt, loss=[loss], metrics=[m])

    save_best = keras.callbacks.ModelCheckpoint(filepath=f'best_model{i}.h5')

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=50, batch_size=32,
                        callbacks=[save_best],
                        shuffle=True, use_multiprocessing=True, verbose=False)
    best_rmse = min(history.history['val_root_mean_squared_error'])
    print(f'RMSE:{best_rmse:.3f} with l:{l}')
    del history

        
        
