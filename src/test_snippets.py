import numpy as np
import keras,gc,nltk
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing
from ssb_vae import *
from util import *

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from gradNormSSBVAE import GradNormSSBVAE

import nltk
nltk.download('wordnet')

name_dat = "Snippets"

tokenizer = TfidfVectorizer().build_tokenizer()
stemmer = SnowballStemmer("english") 
lemmatizer = WordNetLemmatizer()

__random_state__ = 20
np.random.seed(__random_state__)


def read_file(archivo,symb=' '):
    with open(archivo,'r', encoding="utf8") as f:
        lineas = f.readlines()
        tokens_f = [linea.strip().split(symb) for linea in lineas]
        labels = [tokens[-1] for tokens in tokens_f]
        tokens = [' '.join(tokens[:-1]) for tokens in tokens_f]
    return labels,tokens

"""Extract features from raw input"""
def preProcess(s): #String processor
    return s.lower().strip().strip('-').strip('_')

def get_transform_representation(mode, analizer,min_count,max_feat):
    smooth_idf_b = False
    use_idf_b = False
    binary_b = False

    if mode == 'binary':
        binary_b = True
    elif mode == 'tf':     
        pass #default is tf
    elif mode == 'tf-idf':
        use_idf_b = True
        smooth_idf_b = True #inventa 1 conteo imaginario (como priors)--laplace smoothing
    return TfidfVectorizer(stop_words='english',tokenizer=analizer,min_df=min_count, max_df=0.8, max_features=max_feat
                                ,binary=binary_b, use_idf=use_idf_b, smooth_idf=smooth_idf_b,norm=None
                                  ,ngram_range=(1, 3)) 


def run_snippets(percentage_supervision,nbits_for_hashing,alpha_val,lambda_val,beta_VAL,addval=1,reseed=0,seed_to_reseed=20):

    tf.keras.backend.clear_session() 

    labels_t,texts_t = read_file("Data/data-web-snippets/train.txt")
    labels_test,texts_test = read_file("Data/data-web-snippets/test.txt")
    labels = list(set(labels_t))

    labels_t = np.asarray(labels_t)
    labels_test = np.asarray(labels_test)
    texts_train,texts_val,labels_train,labels_val  = train_test_split(texts_t,labels_t,random_state=20,test_size=0.1)

    tokenizer = TfidfVectorizer().build_tokenizer()
    stemmer = SnowballStemmer("english") 
    lemmatizer = WordNetLemmatizer()

    def stemmed_words(doc):
        results = []
        for token in tokenizer(doc):
            pre_pro = preProcess(token)
            #token_pro = stemmer.stem(pre_pro) #aumenta x10 el tiempo de procesamiento
            token_pro = lemmatizer.lemmatize(pre_pro) #so can explain/interpretae -- aumenta x5 el tiempo de proce
            if len(token_pro) > 2 and not token_pro[0].isdigit(): #elimina palabra largo menor a 2
                results.append(token_pro)
        return results

    min_count = 1 #default = 1
    max_feat = 10000 #Best: 10000 -- Hinton (2000)
    vectorizer = get_transform_representation("tf",stemmed_words,min_count,max_feat)

    vectorizer.fit(texts_train)
    vectors_train = vectorizer.transform(texts_train)
    vectors_val = vectorizer.transform(texts_val)
    vectors_test = vectorizer.transform(texts_test)

    token2idx = vectorizer.vocabulary_
    idx2token = {idx:token for token,idx in token2idx.items()}


    #todense --get representation
    X_train = np.asarray(vectors_train.todense())
    X_val = np.asarray(vectors_val.todense())
    X_test = np.asarray(vectors_test.todense())
                          
    del vectors_train,vectors_val,vectors_test#,vectors_train2,vectors_val2,vectors_test2
    gc.collect()

    ##representacion soft para TF ---mucho mejor!
    X_train_input = np.log(X_train+1) 
    X_val_input = np.log(X_val+1) 
    X_test_input = np.log(X_test+1) 

    #outputs as probabolities -- normalized over datasets..
    X_train = X_train/X_train.sum(axis=-1,keepdims=True) 
    X_val = X_val/X_val.sum(axis=-1,keepdims=True)
    X_test = X_test/X_test.sum(axis=-1,keepdims=True)

    X_train[np.isnan(X_train)] = 0
    X_val[np.isnan(X_val)] = 0
    X_test[np.isnan(X_test)] = 0

    X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
    X_total = np.concatenate((X_train,X_val),axis=0)
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

    #Creating and Training the Models

    batch_size = 512

    vae,encoder,generator = SSBVAE(X_train.shape[1],n_classes,Nb=int(nbits_for_hashing),units=500,layers_e=2,layers_d=0,beta=beta_VAL,alpha=alpha_val,lambda_=lambda_val)    # vae.fit(X_total_input, [X_total, Y_total_input], epochs=30, batch_size=batch_size,verbose=1)
    GradNormSSBVAE(vae, X_total_input, [X_total, Y_total_input], [1.0,1.0,1.0], LR=1e-3, alpha=0.12, epochs=100)
    name_model = 'SSB_VAE'

    # toc = time.perf_counter()

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
    #colnames 
    #'dataset', 'algorithm', 'level', 'alpha', 'beta', 'gamma', 'p@100', 'r@100', 'p@1000', 'p@5000', 'map@100', 'map@1000', 'map@5000','added_val_flag','seed_used'
    # file.write("%s, %s, %f, %f, %f, %f, %f, %f, %f,  %f, %f, %f, %f, %d, %d\n"%(name_dat,name_model,percentage_supervision,alpha_val,beta_VAL,gamma_val,p100_b,r100_b,p1000_b,p5000_b,map100_b,map1000_b,map5000_b,addval,seed_to_reseed))
    # file.close()

    del vae, X_total_input, X_total
    del X_train, X_val, X_test
    del total_hash, test_hash
    gc.collect()

    print("DONE ...")

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# from optparse import OptionParser

# op = OptionParser()
# op.add_option("-M", "--model", type=int, default=4, help="model type (1,2,3)")
# op.add_option("-p", "--ps", type=float, default=1.0, help="supervision level (float[0.1,1.0])")
# op.add_option("-a", "--alpha", type=float, default=0.0, help="alpha value")
# op.add_option("-b", "--beta", type=float, default=0.015625, help="beta value")
# op.add_option("-g", "--gamma", type=float, default=0.0, help="gamma value")
# op.add_option("-r", "--repetitions", type=int, default=1, help="repetitions") 
# op.add_option("-o", "--ofilename", type="string", default="results.csv", help="output filename") 
# op.add_option("-s", "--reseed", type=int, default=0, help="if >0 reseed numpy for each repetition") 
# op.add_option("-v", "--addvalidation", type=int, default=1, help="if >0 add the validation set to the train set") 
# op.add_option("-l", "--length_codes", type=int, default=32, help="number of bits") 


# (opts, args) = op.parse_args()

# ps = float(opts.ps)
# tf.keras.backend.set_floatx('float64')

seeds_to_reseed = [20,144,1028,2044,101,6077,621,1981,2806,79]
run_snippets(0.9,16,1.0,1.0,1.0,addval=1,reseed=0,seed_to_reseed=20)
# def run_snippets(percentage_supervision,nbits_for_hashing,alpha_val,lambda_val,beta_VAL,name_file,addval=1,reseed=0,seed_to_reseed=20):

