import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from ssb_vae import *
from utils import *
import gc
import time 
from gradNormSSBVAE import GradNormSSBVAE

name_dat = "20News"

from sklearn.datasets import fetch_20newsgroups
newsgroups_t = fetch_20newsgroups(subset='train')
labels = newsgroups_t.target_names

from utils import Load_Dataset

__random_state__ = 20
np.random.seed(__random_state__)

def run_20_news(percentage_supervision,nbits_for_hashing,alpha_val,addval=1,reseed=0,seed_to_reseed=20, epochs=30, LR=1e-3):

	filename = 'Data/ng20.tfidf.mat'
	data = Load_Dataset(filename)
	X_train_input = data["train"]
	X_train = X_train_input 
	X_val_input = data["cv"]
	X_val = X_val_input 
	X_test_input = data["test"]
	X_test = X_test_input
	labels_train = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_train"]])
	labels_val = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_cv"]])
	labels_test = np.asarray([labels[value.argmax(axis=-1)] for value in data["gnd_test"]])


	#Outputs as probabolities 
	X_train = X_train/X_train.sum(axis=-1,keepdims=True).astype(np.float32)
	X_val = X_val/X_val.sum(axis=-1,keepdims=True).astype(np.float32)
	X_test = X_test/X_test.sum(axis=-1,keepdims=True).astype(np.float32)

	X_train[np.isnan(X_train)] = 0
	X_val[np.isnan(X_val)] = 0
	X_test[np.isnan(X_test)] = 0

	X_total_input = np.concatenate((X_train_input,X_val_input),axis=0).astype(np.float32)
	X_total = np.concatenate((X_train,X_val),axis=0).astype(np.float32)
	labels_total = np.concatenate((labels_train,labels_val),axis=0)
	                        
	#Encoding Labels
	label_encoder = preprocessing.LabelEncoder()
	label_encoder.fit(labels)

	n_classes = len(labels)

	y_train = label_encoder.transform(labels_train)
	y_val = label_encoder.transform(labels_val)
	y_test = label_encoder.transform(labels_test)

	y_train_input = to_categorical(y_train,num_classes=n_classes)
	y_val_input = to_categorical(y_val,num_classes=n_classes)
	y_test_input = to_categorical(y_test,num_classes=n_classes)

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
			y_train_input[idx,:] = np.zeros(n_classes)#hide the label


	Y_total_input = y_train_input

	if addval > 0:#add the validation labels to the train set according to the sup level 

		idx_val = np.arange(0,len(y_val_input),1)
		np.random.shuffle(idx_val)
		np.random.shuffle(idx_val)
		n_sup_val = int(np.floor(percentage_supervision*len(idx_val)))
		idx_sup_val = idx_val[0:n_sup_val]
		idx_unsup_val = idx_val[n_sup_val:]

		if (len(idx_unsup_val) > 0):
			for idx in idx_unsup_val:
				y_val_input[idx,:] = np.zeros(n_classes)#hide the label

		Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)

	
	#Creating and Training the Models

	batch_size = 512

	tf.keras.backend.clear_session()

	tic = time.perf_counter()

	vae,encoder, generator = SSBVAE(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0)
	# vae.fit(X_total_input, [X_total, Y_total_input], epochs=10, batch_size=batch_size,verbose=1)
	name_model = 'SSB_VAE'
	GradNormSSBVAE(vae, X_total_input, [X_total, Y_total_input], [1.0,1.0,1.0], LR=LR, alpha=alpha_val, epochs=epochs)


	toc = time.perf_counter()

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

	del vae, X_total_input, X_total
	del X_train, X_val, X_test
	del total_hash, test_hash
	del data
	gc.collect()

	print("DONE ...")

import os
import sys
from optparse import OptionParser
os.environ["CUDA_VISIBLE_DEVICES"]="1"


op = OptionParser()
# op.add_option("-M", "--model", type=int, default=4, help="model type (1,2,3)")
op.add_option("-p", "--ps"                  , type=float    , default=1.0   , help="supervision level (float[0.1,1.0])")
op.add_option("-a", "--alpha"               , type=float    , default=0.9   , help="alpha value")
op.add_option("-r", "--learning_rate"       , type=float    , default=0.001 , help="learning rate")
op.add_option("-e", "--epochs"              , type=int      , default=150    , help="epochs")
# op.add_option("-a", "--alpha", type=float, default=0.0, help="alpha value")
# op.add_option("-b", "--beta", type=float, default=0.015625, help="beta value")
# op.add_option("-g", "--gamma", type=float, default=0.0, help="gamma value")
# op.add_option("-r", "--repetitions", type=int, default=1, help="repetitions") 
# op.add_option("-o", "--ofilename", type="string", default="results.csv", help="output filename") 
# op.add_option("-s", "--reseed", type=int, default=0, help="if >0 reseed numpy for each repetition") 
# op.add_option("-v", "--addvalidation", type=int, default=1, help="if >0 add the validation set to the train set") 
op.add_option("-l", "--length_codes"        , type=int      , default=32    , help="number of bits") 


(opts, args) = op.parse_args()
ps = float(opts.ps)
l = int(opts.length_codes)
a = float(opts.alpha)
LR = float(opts.learning_rate)
e = int(opts.epochs)

run_20_news(ps,l,a, LR=LR, epochs=e)