# import libraries and modules
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
import tensorflow as tf
from keras import regularizers
import pandas as pd
import os
import wandb
from wandb.keras import WandbCallback
import argparse
from pathlib import Path
import json

wandb.login(key='24105b2510470e8d99c2a27e9d5c53f3934aa355')
wandb.init(project="Antenna_model", entity="vojta-bednarsky")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epo', type=int, default=250, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=256, help='Batch size')
    parser.add_argument('--opt', type=str, default='Adamax', help='Optimizer')
    parser.add_argument('--m', type=float, default=0.0, help='Momentum for SDG optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay')
    parser.add_argument('--af', type=str, default='relu', help='Activation function for all layers')
    parser.add_argument('--af2', type=str, default='relu', help='Activation function for all layers')
    parser.add_argument('--u', type=int, default=64, help='Number of neurons in first hidden layer')
    parser.add_argument('--u2', type=int, default=64, help='Number of neurons in second hidden layer')
    config = vars(parser.parse_args())

    run_name = 'run'
    save_dir = Path('training')
    if not save_dir.exists():
        os.mkdir(save_dir)
    save_dir = save_dir / run_name
    if save_dir.exists():
        for i in range(len(next(os.walk('training'))[1])):
            save_dir = Path('training') / Path(run_name + '_' + str(i))
            if not save_dir.exists():
                break
    os.mkdir(save_dir)

    with open(save_dir / Path('commandline_args.txt'), 'w') as f:
        json.dump(parser.parse_args().__dict__, f, indent=2)

    # set the learning rate for the optimizer
    lr = config['lr']
    epo = config['epo']
    bs = config['bs']
    opt_name = config['opt']
    m = config['m']
    lr_decay = config['lr_decay']
    af = config['af']
    u = config['u']
    u2 = config['u2']
    wandb.log(dict(config))

    if opt_name == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=m)
    elif opt_name == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=lr)
    elif opt_name == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'Adamax':
        opt = keras.optimizers.Adamax(learning_rate=lr)
    else:
        print(f'Incorrect optimizer names')
        exit()


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

    """ 
    Seznam x obsahuje hodnoty, které se použijí jako počet uzlů v první husté vrstvě modelu.
    Pro každou hodnotu l v tomto seznamu se vytvoří nový model s první hustou vrstvou o velikosti l]  
    """

    # create a sequential model
    model = keras.Sequential()
    # add an input layer with 2 input nodes
    model.add(Input(shape=(2,)))
    # add a dense layer with l nodes and ReLU activation function
    # for _ in range(layers):
    model.add(Dense(units=u, activation=af, kernel_initializer=initializer))
    # add another dense layer with 200 nodes and ReLU activation function
    model.add(Dense(units=u2, activation='relu', kernel_initializer=initializer))
              #       kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
              #       bias_regularizer=regularizers.L2(1e-4),
              #       activity_regularizer=regularizers.L2(1e-5)))
    # add a dropout layer to prevent overfitting
    model.add(Dropout(.10))
    # add a dense output layer with 100 nodes and softmax activation function
    model.add(Dense(units=100, activation='sigmoid', kernel_initializer=initializer))

    # create an Adam optimizer with the specified learning rate
    opt = keras.optimizers.Adam(learning_rate=lr)
    # create a root mean squared error metric to evaluate the model performance
    m = keras.metrics.RootMeanSquaredError()
    # create a categorical crossentropy loss function to train the model
    loss = keras.losses.CategoricalCrossentropy()

    # compile the model with the specified optimizer, loss function and metric
    model.compile(optimizer=opt, loss=[loss], metrics=[m])

    # create a callback to save the best performing model during training
    save_best = keras.callbacks.ModelCheckpoint(filepath=f'{save_dir}/best_model.h5')


    # train the model on the training data and evaluate it on the validation data
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epo, batch_size=bs,
                        callbacks=[save_best,WandbCallback(save_model=False)],
                        shuffle=True, use_multiprocessing=True, verbose=False)

    # find the minimum root mean squared error on the validation data during training
    best_rmse = min(history.history['val_root_mean_squared_error'])

    # print the best root mean squared error and the number of nodes in the first dense layer (l)
    print(f'RMSE:{best_rmse:.3f} with L-{l}')
    wandb.log({"val_RMSE": best_rmse})
