import tensorflow as tf

import random
import numpy as np
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from gradNorm import GradNorm

seed = 40
random.seed(seed)

# Data to create Toy Example of Paper
def createData(B, epsilons, sigmas, T=2, samples=10000):
    X = np.random.uniform(0, 1, size=(samples, 250)).astype(np.float32)
    ys = []
    for i in range(samples):
        x = X[i]
        y = []
        for j in range(T):
            y.append(sigmas[j]*np.tanh((B + epsilons[j]).dot(x)))
        ys.append(y)
    return X, np.array(ys)     

# Toy example model
def toyExample(input_dim=250, layers=4, neurons=100, activation='relu', Tasks=2):
    x = Input(shape = (input_dim, ))
    shared = Dense(neurons)(x)
    for i in range(layers-1):
        if i == layers - 2:
            shared = Dense(neurons, activation=activation, name='last_shared_layer')(shared)
        else:
            shared = Dense(neurons, activation=activation)(shared)
    outputs = []
    for i in range(Tasks):
        outputs.append(Dense(100, activation='linear', name=f'task-{i}')(shared))
    
    model = Model(inputs= x, outputs = outputs)
    # model.compile(loss='mse', optimizer=SGD(lr=0.001), metrics=['accuracy'])
    return model


def main():
    Tasks = 2

    B = np.random.normal(scale=10, size=(100, 250)).astype(np.float32)
    epsilons = np.random.normal(scale=3.5, size=(Tasks, 100, 250)).astype(np.float32)
    sigmas = [1.0, 100.0]

    X, Y = createData(B, epsilons, sigmas, T = Tasks)
    model = toyExample()
    # model.summary()

    losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()]
    metrics = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
    weights = [1.0, 1.0]
    # model = toyExample()
    tf.keras.backend.clear_session()
    model=toyExample()

    Ls, Ws = GradNorm(model, X, Y, 2, weights, losses, metrics, LR=1e-3, batch_size=128, epochs=400, verbose=True)
    print("Steps: ", len(Ls[0]))


if __name__ == "__main__":
    main()

