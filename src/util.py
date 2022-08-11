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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from gradNormSSBVAE import GradNormSSBVAE

name_dat = "CIFAR-10"
__random_state__ = 20

def evaluate_hashing_DE(labels,train_hash,test_hash,labels_trainn,labels_testt,tipo="topK",eval_tipo='PRatk',K=100):
    """
        Evaluate Hashing correclty: Query and retrieve on a different set
    """
    test_similares_train =  get_similar(test_hash,train_hash,tipo=tipo,K=K)
    if eval_tipo=="MAP":
        return MAP_atk(test_similares_train,labels_query=labels_testt, labels_source=labels_trainn, K=K) 
    elif eval_tipo == "PRatk":
        return measure_metrics(labels,test_similares_train,labels_testt,labels_source=labels_trainn)
    elif eval_tipo == "Patk":
        return M_P_atk(test_similares_train, labels_query=labels_testt, labels_source=labels_trainn, K=K)

def hash_data(model, x_train, x_test, binary=True):
    encode_train = model.predict(x_train)
    encode_test = model.predict(x_test)
    
    train_hash = calculate_hash(encode_train, from_probas=binary )
    test_hash = calculate_hash(encode_test, from_probas = binary)
    return train_hash, test_hash

def define_fit(multi_label,X,Y, epochs=20, dense_=True):
    #function to define and train model

    #define model
    model_FF = Sequential()
    model_FF.add(InputLayer(input_shape=(X.shape[1],) ))
    if dense_:
        model_FF.add(Dense(256, activation="relu"))
    #model_FF.add(Dense(128, activation="relu"))
    if multi_label:
        model_FF.add(Dense(Y.shape[1], activation="sigmoid"))
        model_FF.compile(optimizer='adam', loss="binary_crossentropy")
    else:
        model_FF.add(Dense(Y.shape[1], activation="softmax"))
        model_FF.compile(optimizer='adam', loss="categorical_crossentropy",metrics=["accuracy"])
    model_FF.fit(X, Y, epochs=epochs, batch_size=128, verbose=0)
    return model_FF


class MedianHashing(object):
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape, dtype='int32')
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def calculate_hash(data, from_probas=True, from_logits=True):    
    if from_probas: #from probas
        if from_logits:
            from scipy.special import expit
            data = expit(data)
        data_hash = (data > 0.5)*1
    else: #continuos
        data_hash = (np.sign(data) + 1)/2
    return data_hash.astype('int32')

def get_hammD(query, corpus):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_hammD = np.zeros((len(query),len(corpus)),dtype='int16') #distancia no sera mayor a 2^16
    for i,dato_hash in enumerate(query):
        query_hammD[i] = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
    return query_hammD

def get_similar_hammD_based(query_hammD,tipo="topK", K=100, ball=0):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    query_similares = [] #indices
    for i in range(len(query_hammD)):        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(query_hammD[i] <= ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(query_hammD[i]) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares


def xor(a,b):
    return (a|b) & ~(a&b)
def calculate_hamming_D(a,B):
    #return np.sum(a.astype('bool')^ B.astype('bool') ,axis=1) #distancia de hamming (# bits distintos)
    #return np.sum(np.logical_xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    v = np.sum(a != B,axis=1) #distancia de hamming (# bits distintos) -- fastest
    #return np.sum(xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    return v.astype(a.dtype)

def get_similar(query, corpus,tipo="topK", K=100, ball=2):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_similares = [] #indices
    for dato_hash in query:
        hamming_distance = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
        if tipo=="EM": #match exacto
            ball= 0
        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(hamming_distance<=ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(hamming_distance) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares

def measure_metrics(labels_name, data_retrieved_query, labels_query, labels_source):
    """
        Measure precision at K and recall at K, where K is the len of the retrieval documents
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
        
    multi_label=False
    if type(labels_query[0]) == list or type(labels_query[0]) == np.ndarray: #multiple classes
        multi_label=True
    
    #relevant document for query data
    
    if multi_label:
        count_labels = {label: np.sum([label in aux for aux in labels_source]) for label in labels_name}
    else:
        count_labels = {label: np.sum([label == aux for aux in labels_source]) for label in labels_name}
    
    #count_labels = {label:np.sum([label in aux for aux in labels_source]) for label in labels_name} 
    
    precision = 0.
    recall =0.
    for similars, label in zip(data_retrieved_query, labels_query): #source de donde se extrajo info
        if len(similars) == 0: #no encontro similares:
            continue
        labels_retrieve = labels_source[similars] #labels of retrieved data
        
        if multi_label: #multiple classes
            tp = np.sum([len(set(label)& set(aux))>=1 for aux in labels_retrieve]) #al menos 1 clase en comun --quizas variar
            recall += tp/np.sum([count_labels[aux] for aux in label ]) #cuenta todos los label del dato
        else: #only one class
            tp = np.sum(labels_retrieve == label) #true positive
            recall += tp/count_labels[label]
        precision += tp/len(similars)
    
    return precision/len(labels_query), recall/len(labels_query)

def P_atk(labels_retrieved, label_query, K=1):
    """
        Measure precision at K
    """
    if len(labels_retrieved)>K:
        labels_retrieved = labels_retrieved[:K]

        
    if type(labels_retrieved[0]) == list or type(labels_retrieved[0]) == np.ndarray: #multiple classes
        tp = np.sum([len(set(label_query)& set(aux))>=1 for aux in labels_retrieved]) #al menos 1 clase en comun --quizas variar
    else: #only one class
        tp = np.sum(labels_retrieved == label_query) #true positive
    
    return tp/len(labels_retrieved) #or K

def M_P_atk(datas_similars, labels_query, labels_source, K=1):
    """
        Mean (overall the queries) precision at K
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
    return np.mean([P_atk(labels_source[datas_similars[i]],labels_query[i],K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)])


def AP_atk(data_retrieved_query, label_query, labels_source, K=0):
    """
        Average precision at K, average all the list precision until K.
    """
    multi_label=False
    if type(label_query) == list or type(label_query) == np.ndarray: #multiple classes
        multi_label=True
        
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
        
    if K == 0:
        K = len(data_retrieved_query)
    
    K_effective = K
    if len(data_retrieved_query) < K:
        K_effective = len(data_retrieved_query)
    elif len(data_retrieved_query) > K:
        data_retrieved_query = data_retrieved_query[:K]
        K_effective = K
    
    labels_retrieve = labels_source[data_retrieved_query] 
    
    score = []
    num_hits = 0.
    for i in range(K_effective):
        relevant=False
        
        if multi_label:
            if len( set(label_query)& set(labels_retrieve[i]) )>=1: #at least one label in comoon at k
                relevant=True
        else:
            if label_query == labels_retrieve[i]: #only if "i"-element is relevant 
                relevant=True
        
        if relevant:
            num_hits +=1 
            score.append(num_hits/(i+1)) #precition at k 

    if len(score) ==0:
        return 0
    else:
        return np.mean(score) #average all the precisions until K

def MAP_atk(datas_similars, labels_query, labels_source, K=0):
    """
        Mean (overall the queries) average precision at K
    """
    return np.mean([AP_atk(datas_similars[i], labels_query[i], labels_source, K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)]) 

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
    # std = MinMaxScaler()
    std.fit(X_t)

    X_t = std.transform(X_t)
    X_test = std.transform(X_test)

    X_train, X_val, labels_train, labels_val  = train_test_split(X_t, labels_t, random_state=20, test_size=len(X_test))

    del X_t, labels_t
    gc.collect()

    X_train_input = X_train
    X_val_input = X_val
    X_test_input = X_test

    X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)#.astype(np.float64)
    X_total = np.concatenate((X_train,X_val),axis=0)#.astype(np.float64)
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

        Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)#.astype(np.float64)

    return n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input


# # tf.keras.backend.set_floatx('float64')


seeds_to_reseed = [20,144,1028,2044,101,6077,621,1981,2806,79]
batch_size = 100*2
tf.keras.backend.clear_session()
tic = time.perf_counter()
n_classes, labels, labels_total, labels_test, X_total, X_test, X_total_input, X_test_input, Y_total_input = load_data(0.9,addval=1,reseed=0,seed_to_reseed=79)
vae,encoder,generator = SSBVAE(X_total.shape[1],n_classes,Nb=int(16),units=500,layers_e=2,layers_d=0,beta=10000000.0 ,alpha=10000.0,lambda_=0.015625)

print(X_total_input.shape)
print(Y_total_input.shape)

print(X_test_input)

# Y = np.concatenate([X_total, Y_total_input], axis=1)

# print(Y.shape)

from tensorflow.keras.utils import plot_model
plot_model(vae, show_shapes=True)

# vae.fit(X_total_input, [X_total, Y_total_input], epochs=10 , batch_size=batch_size,verbose=1)
GradNormSSBVAE(vae, X_total_input, [X_total, Y_total_input], [1.0, 1.0, 1.0], 
               verbose=True, epochs=40, gradNorm=True, alpha=0.5, LR=1e-3, batch_size=512)
# GradNormSSBVAE(vae, X_total_input, [X_total, Y_total_input], 2, [1.0, 1.0, 1.0], [True, True, True], losses, losses, verbose=True, epochs=40,gradNorm=True, alpha=1.5, LR=1e-1, batch_size=128)

total_hash, test_hash = hash_data(encoder,X_total_input,X_test_input)

p100_b,r100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK")
map100_b = evaluate_hashing_DE(labels,total_hash, test_hash,labels_total,labels_test,tipo="topK",eval_tipo="MAP",K=100)
print("p100_b: ", p100_b)
print("r100_b: ", r100_b)
print("map100_b: ", map100_b)
