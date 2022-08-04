import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from ssb_vae import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
# from utils import sample_test_mask
from sklearn.preprocessing import StandardScaler
import time

name_dat = "CIFAR-10"
__random_state__ = 20


def enmask_data(data, mask):
    if type(data) == list:
        return np.asarray(data)[mask].tolist()
    elif type(data) == np.ndarray:
        return data[mask]

def sample_test_mask(labels_list, N=100, multi_label=True):
    idx_class = {}
    for value in np.arange(len(labels_list)):
        if multi_label:
            for tag in labels_list[value]:
                if tag in idx_class:
                    idx_class[tag].append(value)
                else:
                    idx_class[tag] = [value]
        else:
            tag = labels_list[value]
            if tag in idx_class:
                idx_class[tag].append(value)
            else:
                idx_class[tag] = [value]

    mask_train = np.ones(len(labels_list), dtype='bool')
    selected = []
    for clase in idx_class.keys():
        selected_clase = []
        for dato in idx_class[clase]:
            if dato not in selected:
                selected_clase.append(dato) # si dato no ha sido seleccionado como rep de otra clase se guarda

        v = np.random.choice(selected_clase, size=N, replace=False)
        selected += list(v)
        mask_train[v] = False #test set
    return mask_train

def load_data(percentage_supervision,addval=1,reseed=0,seed_to_reseed=20):
    
    (_, aux_t), (_, aux_test) = keras.datasets.cifar10.load_data()

    labels = ["airplane", "automobile","bird", "cat","deer","dog","frog","horse","ship","truck"]
    labels_t = np.asarray([labels[value[0]] for value in aux_t])
    labels_test = np.asarray([labels[value[0]] for value in aux_test])
    labels_t = np.concatenate((labels_t,labels_test),axis=0)

    X_t = np.load("Data/cifar10_VGG_avg.npy") #mejora
    X_t.shape

    mask_train = sample_test_mask(labels_t, N=100)

    X_test = X_t[~mask_train]
    X_t = X_t[mask_train]
    labels_test = enmask_data(labels_t, ~mask_train)
    labels_t = enmask_data(labels_t, mask_train)

    gc.collect()

    std = StandardScaler(with_mean=True, with_std=True)
    std.fit(X_t)

    X_t = std.transform(X_t)
    X_test = std.transform(X_test)

    X_train, X_val, labels_train, labels_val  = train_test_split(X_t, labels_t, random_state=20, test_size=len(X_test))

    del X_t, labels_t
    gc.collect()

    X_train_input = X_train
    X_val_input = X_val
    X_test_input = X_test

    X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
    X_total = np.concatenate((X_train,X_val),axis=0)
    labels_total = np.concatenate((labels_train,labels_val),axis=0)

    #print("\n=====> Encoding Labels ...\n")
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)

    n_classes = len(labels)

    y_train = label_encoder.transform(labels_train)
    y_val = label_encoder.transform(labels_val)
    y_test = label_encoder.transform(labels_test)

    y_train_input = to_categorical(y_train,num_classes=n_classes)
    y_val_input = to_categorical(y_val,num_classes=n_classes)
    y_test_input = to_categorical(y_test,num_classes=n_classes)

    ##RESEED?
    if reseed > 0:
        np.random.seed(seed_to_reseed)
    else:
        np.random.seed(__random_state__)

    idx_train = np.arange(0,len(y_train_input),1)
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_train)
    n_sup = int(np.floor(percentage_supervision*len(idx_train)))
    idx_sup = idx_train[0:n_sup]
    idx_unsup = idx_train[n_sup:]

    if (len(idx_unsup) > 0):
        for idx in idx_unsup:
            y_train_input[idx,:] = np.zeros(n_classes)

    Y_total_input = y_train_input

    if addval > 0:

        idx_val = np.arange(0,len(y_val_input),1)
        np.random.shuffle(idx_val)
        np.random.shuffle(idx_val)
        n_sup_val = int(np.floor(percentage_supervision*len(idx_val)))
        idx_sup_val = idx_val[0:n_sup_val]
        idx_unsup_val = idx_val[n_sup_val:]

        if (len(idx_unsup_val) > 0):
            for idx in idx_unsup_val:
                y_val_input[idx,:] = np.zeros(n_classes)

        Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)

    return n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input

seeds_to_reseed = [20,144,1028,2044,101,6077,621,1981,2806,79]
batch_size = 100*2
tf.keras.backend.clear_session()
tic = time.perf_counter()
n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input = load_data(0.1,addval=1,reseed=0,seed_to_reseed=20)
vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(32),units=500,layers_e=2,layers_d=0,beta=0.003906 ,alpha=100000.0,gamma=1000000.0)

vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)