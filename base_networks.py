from unicodedata import name
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


def mean_KL_loss(z_mean,z_log_var):
    #the mean and log-var of the latent distribution
    def KL(y_true, y_pred):
        return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) #con varianza
    return KL

def KL_loss(z_mean,z_log_var):
    #the mean and log-var of the latent distribution
    def KL(y_true, y_pred):
        return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) #con varianza
    return KL

def KrossEntropy(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred)

def mean_BKL_loss():
    p_b = keras.activations.sigmoid(logits_b) #B_j = Q(b_j) probability of b_j
    Nb = K.int_shape(p_b)[1]
    ep = K.epsilon()
    def KL(y_true, y_pred):
        return np.log(2) + K.mean( p_b*K.log(p_b + ep) + (1-p_b)* K.log(1-p_b +ep),axis=1)
    return KL

def BKL_loss(logits_b):
    p_b = keras.activations.sigmoid(logits_b) #B_j = Q(b_j) probability of b_j
    Nb = K.int_shape(p_b)[1]
    ep = K.epsilon()
    def KL(y_true, y_pred):
        return (Nb*np.log(2) + K.sum( p_b*K.log(p_b + ep) + (1-p_b)* K.log(1-p_b +ep),axis=1))
    return KL

class Beta_Call(keras.callbacks.Callback):   
    def __init__(self, beta_ann, kl_inc= 1./5000, max_KL=0.1, verbose=0):
        #default parameters for text datasets..
        self.beta_ann = beta_ann
        self.kl_inc = kl_inc
        self.max_KL = max_KL
        self.verbose = verbose
        super(Beta_Call,self).__init__()
        
    def on_epoch_end(self, epoch, logs={}):    
        K.set_value(self.beta_ann, np.min([K.get_value(self.beta_ann)+self.kl_inc*(epoch+1), self.max_KL])) 
        if self.verbose==1:
            print("Epoch",epoch,"the KL weight is",K.get_value(self.beta_ann)) 


def define_pre_encoder(data_dim,layers=2,units=512,dropout=0.0,BN=False): #define pre_encoder network
    model = Sequential(name='pre-encoder')
    model.add(InputLayer(input_shape=(data_dim,)))
    for i in range(1,layers+1):
        #model.add(Dense(int(units/i), activation='relu'))
        
        if i == layers:
            model.add(Dense(units,activation='relu', name='last_shared_layer'))
        else:
            model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    return model

def define_generator(Nb,data_dim,layers=2,units=32,dropout=0.0,BN=False,out_type='softmax'):
    model = Sequential(name='generator/decoder')
    model.add(InputLayer(input_shape=(Nb,)))
    for i in np.arange(layers,0,-1):
        #model.add(Dense(int(units/i), activation='relu'))
        model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    #if exclusive:
    model.add(Dense(data_dim, activation=out_type)) #softmax generator
    #else:
    #    model.add(Dense(data_dim, activation='sigmoid'))
    return model

def add_Conv(it, filters, kernel_s, BN = False, **args):
    f1 = Conv2D(filters, kernel_s, padding='same', **args)(it)
    if BN:
        f1 = BatchNormalization()(f1)
    return f1                                            
def conv_bloq(it, filters, kernel_s, max_pool=0, BN=False,double=False, **args):
    f1 = add_Conv(it, filters, kernel_s, BN = BN, **args)
    if double:
        f1 = add_Conv(f1, filters, kernel_s, BN = BN, **args)
        
    if max_pool!= 0:
        f1 = MaxPool2D(max_pool)(f1)
    return f1

def def_pre_encoder_CNN(input_dim, kernel_s, L=1, filters=32, max_pool=0, BN=False, double=False,dense_=False, **args): 
    it = Input(shape=input_dim)  #fixed length..
    f1 = it
    for l in range(L):
        f1 = conv_bloq(f1, filters, kernel_s, max_pool=max_pool, BN=BN, double=double, **args)         
        filters = int(filters*2)
        
    shape_before_F = K.int_shape(f1)[1:] 
    out_x = Flatten()(f1)
    
    if dense_:
        out_x = Dense(128, activation='relu')(out_x) 
        if BN:
            out_x = BatchNormalization()(out_x)
    return Model(inputs=it, outputs=out_x, name='pre-encoder'), shape_before_F


def add_ConvT(it, filters, kernel_s, BN = False, **args):
    f1 = Conv2DTranspose(filters, kernel_s, padding='same', **args)(it)
    if BN:
        f1 = BatchNormalization()(f1)
    return f1
def convT_bloq(it, filters, kernel_s, max_pool=0, BN=False,double=False, **args):
    f1 = add_ConvT(it, filters, kernel_s, BN = BN, **args)
    if double:
        f1 = add_ConvT(f1, filters, kernel_s, BN = BN, **args)
        
    if max_pool!= 0:
        f1 = UpSampling2D(max_pool)(f1)
    return f1


def define_generator_CNN(shape_before_F, kernel_s, L=1, filters=32, max_pool=0, BN=False, double=False,out_shape =[], dense_=False,   **args): 
    it = Input(shape=(1,), name="dummy_inp")  #fixed length..
    f1 = it
    
    if dense_:
        f1 = Dense(128, activation='relu')(f1)
        #if BN:
        #    f1 = BatchNormalization()(f1)
    
    f1 = Dense(np.prod(shape_before_F), activation='linear')(f1) 
    
    f1 = Reshape(shape_before_F)(f1)
    #if BN:
    #    f1 = BatchNormalization()(f1)
    
    filters = int(filters*2**(L-1))
    for l in range(L):
        f1 = convT_bloq(f1, filters, kernel_s, max_pool=max_pool, BN=BN, double=double, **args)         
        filters = int(filters/2)
    
    channels = 1
    if len(out_shape) !=0:
        channels = out_shape[-1]
    out_x = Conv2D(channels, kernel_s, strides=1, padding='same', activation='sigmoid')(f1)

    #check reconstructed data shape vs needed recosntructed shape
    if len(out_shape) !=0:
        _, d_x, d_y,_ = K.int_shape(out_x)
        delta_x = out_shape[0] - d_x
        delta_y = out_shape[1] - d_y

        padd_len_x = int(np.abs(delta_x/2)) #la mitad en cad alado
        if np.abs(delta_x) % 2 !=0:
            padd_len_x += 1

        padd_len_y = int(np.abs(delta_y/2)) #la mitad en cad alado
        if np.abs(delta_y) % 2 !=0:
            padd_len_y += 1              
        if delta_x > 0 or delta_y > 0:
            out_x = ZeroPadding2D((padd_len_x, padd_len_y))(out_x) #fill con zeros
        elif delta_x < 0 or delta_y < 0: 
            out_x = Cropping2D((padd_len_x, padd_len_y))(out_x)

    return Model(inputs=it, outputs=out_x, name='generator/decoder')

def samp_gumb(logits, tau=0.67):
    from scipy.special import expit    
    eps = 1e-7
    U = np.random.uniform(0, 1, logits.shape)
    b = logits + np.log(U + eps)- np.log(1-U + eps)
    return expit(b/tau) 


def Hamming_loss(y_true, y_pred, b_sampled, Nb=16):
        
        #pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        r = tf.reduce_sum(b_sampled*b_sampled, 1)
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.linalg.matmul(b_sampled, tf.transpose(b_sampled)) + tf.transpose(r) #BXB
     
        similar_mask = K.dot(y_pred, K.transpose(y_pred)) #BXB  M_ij = I(y_i = y_j)  
        loss_hamming = (1.0/Nb)*K.sum(similar_mask*D + (1.0-similar_mask)*K.relu((Nb/3.0)-D))

        # return beta*pred_loss(y_true, y_pred) + alpha*loss_hamming
        return loss_hamming