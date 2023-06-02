# import libraries and modules
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
import tensorflow as tf
from keras import regularizers
import pandas as pd
import os

# visibility of the settings weights and more other parameters
#https://netron.app/

# import PDE dataset
# [import datasetu + nutnost oddělovače]
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

# initialize the weights of the neural network with random values
# [inicializují se váhy neuronové sítě náhodnými hodnotami]
initializer = keras.initializers.RandomUniform()
rmse = 99999

""" 
Seznam x obsahuje hodnoty, které se použijí jako počet uzlů v první husté vrstvě modelu.
Pro každou hodnotu l v tomto seznamu se vytvoří nový model s první hustou vrstvou o velikosti l]  
""" 

# [Pro každou hodnotu l v seznamu se vytvoří sekvenční model neuronové sítě s několika vrstvami]
x = [x for x in range(10,600,100)]
for i, l  in enumerate(x) :
     
    # create a sequential model
    model = keras.Sequential()
    # add an input layer with 2 input nodes
    model.add(Input(shape=(2,)))
    # add a dense layer with l nodes and ReLU activation function
    model.add(Dense(units=l, activation='relu', kernel_initializer=initializer))
    # add another dense layer with 200 nodes and ReLU activation function
    model.add(Dense(units=10, activation='relu', kernel_initializer=initializer))    
              #       kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
              #       bias_regularizer=regularizers.L2(1e-4),
              #       activity_regularizer=regularizers.L2(1e-5)))
    # add a dropout layer to prevent overfitting
    model.add(Dropout(.10))
    # add a dense output layer with 100 nodes and softmax activation function
    model.add(Dense(units=100, activation='softmax', kernel_initializer=initializer))
    
    # set the learning rate for the optimizer
    lr = 0.05
    # create an Adam optimizer with the specified learning rate
    opt = keras.optimizers.Adam(learning_rate=lr)
    # create a root mean squared error metric to evaluate the model performance
    m = keras.metrics.RootMeanSquaredError()
    # create a categorical crossentropy loss function to train the model
    loss = keras.losses.CategoricalCrossentropy()

    # compile the model with the specified optimizer, loss function and metric
    model.compile(optimizer=opt, loss=[loss], metrics=[m])

    # create a callback to save the best performing model during training
    save_best = keras.callbacks.ModelCheckpoint(filepath=f'best_model{i}_L{l}.h5')


    # train the model on the training data and evaluate it on the validation data
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=50, batch_size=32,
                        callbacks=[save_best],
                        shuffle=True, use_multiprocessing=True, verbose=False)
    
    # find the minimum root mean squared error on the validation data during training
    best_rmse = min(history.history['val_root_mean_squared_error'])
    
    # print the best root mean squared error and the number of nodes in the first dense layer (l)
    print(f'RMSE:{best_rmse:.3f} with L-{l}')
    
    # delete the history object to free up memory
    del history