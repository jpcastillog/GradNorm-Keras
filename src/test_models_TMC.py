import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from ssb_vae import *
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from keras.utils import to_categorical
from sklearn import preprocessing
from gradNormSSBVAE import GradNormSSBVAE

name_dat = "TMC"

__random_state__ = 20
np.random.seed(__random_state__)


def load_data(percentage_supervision,addval=1,reseed=0,seed_to_reseed=20):

    labels = ['a','b','c','d','e','f','e','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']

    filename = 'Data/tmc.tfidf.mat'
    data = Load_Dataset(filename)

    X_train_input = np.array(data["train"],dtype=np.float32)
    X_train = X_train_input 
    X_val_input = np.array(data["cv"],dtype=np.float32)
    X_val = X_val_input 
    X_test_input = np.array(data["test"],dtype=np.float32)
    X_test = X_test_input

    labels = np.asarray(labels)
    labels_train = np.asarray([labels[value.astype(bool)] for value in data["gnd_train"]])
    labels_val = np.asarray([labels[value.astype(bool)] for value in data["gnd_cv"]])
    labels_test = np.asarray([labels[value.astype(bool)] for value in data["gnd_test"]])

    del data
    gc.collect()

    #outputs as probabolities -- normalized over datasets..
    X_train = X_train/X_train.sum(axis=-1,keepdims=True) 
    X_val = X_val/X_val.sum(axis=-1,keepdims=True)
    X_test = X_test/X_test.sum(axis=-1,keepdims=True)

    X_train[np.isnan(X_train)] = 0
    X_val[np.isnan(X_val)] = 0
    X_test[np.isnan(X_test)] = 0

    X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
    X_total = np.concatenate((X_train,X_val),axis=0)
    print(X_total_input)
    
    labels_total = np.concatenate((labels_train,labels_val),axis=0)
    labels_full = np.concatenate((labels_total,labels_test),axis=0)

    #Encoding Labels

    n_classes = len(labels)

    label_encoder = preprocessing.MultiLabelBinarizer() 
    label_encoder.fit(labels_full)

    n_classes = len(label_encoder.classes_)

    y_train_input = label_encoder.transform(labels_train)
    y_val_input = label_encoder.transform(labels_val)
    y_test_input = label_encoder.transform(labels_test)

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
            y_train_input[idx,:] = np.zeros(n_classes)#hide the labels

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
                y_val_input[idx,:] = np.zeros(n_classes)#hide the labels

        Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)

    return n_classes, labels, labels_total, labels_test, X_total.astype(np.float32), X_test.astype(np.float32), X_total_input.astype(np.float32), X_test_input.astype(np.float32), Y_total_input.astype(np.float32)

#MODIFICA ESEGUITA
#Vecchia versione: def run_TMC(model_id,percentage_supervision,nbits_for_hashing,alpha_val,gamma_val,beta_VAL,name_file, addval,reseed,seed_to_reseed, n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input):
#Sostituisci gamma_val con lambda_val
def run_TMC(percentage_supervision,nbits_for_hashing,alpha_val,lambda_val,beta_VAL, addval,reseed,seed_to_reseed, n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input):
 
    #Creating and Training the Models 

    tf.keras.backend.clear_session()

    batch_size = 512



    #MODIFICA ESEGUITA
    #Vecchia versione: vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,gamma=gamma_val)
    #Sostituisci gamma con lambda_ , e gamma_val con lambda_val
    vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,lambda_=lambda_val)
    # vae.fit(X_total_input, [X_total, Y_total_input], epochs=10, batch_size=batch_size,verbose=1)
    GradNormSSBVAE(vae, X_total_input, [X_total, Y_total_input], [1.0,1.0,1.0], LR=1e-3, alpha=0.12, epochs=30, batch_size=512)

    name_model = 'SSB_VAE'


    print("\n=====> Evaluate the Models ... \n")

    total_hash, test_hash = hash_data(encoder,X_total_input,X_test_input)
    

    p100_b,r100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK")
    p5000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=5000)
    p1000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="Patk",K=1000)
    map5000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=5000)
    map1000_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=1000)
    map100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=100)

    print("p100_b: ", p100_b)
    print("r100_b: ", r100_b)
    print("map100_b: ", map100_b)

    #colnames  - Modifica Eseguita al commento: Sostituitoa gamma con lambda_
    #'dataset', 'algorithm', 'level', 'alpha', 'beta', 'lambda_', 'p@100', 'r@100', 'p@1000', 'p@5000', 'map@100', 'map@1000', 'map@5000','added_val_flag','seed_used'
    #MODIFICA ESEGUITA
    #Vecchia versione: file.write("%s, %s, %f, %f, %f, %f, %f, %f, %f,  %f, %f, %f, %f, %d, %d\n"%(name_dat,name_model,percentage_supervision,alpha_val,beta_VAL,gamma_val,p100_b,r100_b,p1000_b,p5000_b,map100_b,map1000_b,map5000_b,addval,seed_to_reseed))
    #Sostituisci gamma_val con lambda_val


    del vae, total_hash, test_hash
    gc.collect()

    print("DONE ...")

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# tf.keras.backend.set_floatx('float32')

#from optparse import OptionParser

#op = OptionParser()
#op.add_option("-M", "--model", type=int, default=4, help="model type (1,2,3)")
#op.add_option("-p", "--ps", type=float, default=1.0, help="supervision level (float[0.1,1.0])")
#op.add_option("-a", "--alpha", type=float, default=0.0, help="alpha value")
#op.add_option("-b", "--beta", type=float, default=0.000244, help="beta value")
#op.add_option("-l", "--lambda_", type=float, default=0.0, help="lambda value")
#op.add_option("-r", "--repetitions", type=int, default=1, help="repetitions")
#op.add_option("-o", "--ofilename", type="string", default="results.csv", help="output filename")
#op.add_option("-s", "--reseed", type=int, default=0, help="if >0 reseed numpy for each repetition")
#op.add_option("-v", "--addvalidation", type=int, default=1, help="if >0 add the validation set to the train set")
#op.add_option("-c", "--length_codes", type=int, default=32, help="number of bits")

#(opts, args) = op.parse_args()
ps=0.9
n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input = load_data(ps,addval=1,reseed=0,seed_to_reseed=20)
run_TMC(ps,16,1,1,1,1,0,20,n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input)

