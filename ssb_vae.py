import numpy as np
import keras
from keras.layers import *
from keras.models import Sequential,Model
from tensorflow.keras import backend as K
from base_networks import *
import tensorflow as tf

class SamplingLayer(tf.keras.layers.Layer):
    def __init__(self, tau, units) :
        super(SamplingLayer, self).__init__()
        self.tau = K.variable(tau)
        self.units = units
    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                dtype='float32'),trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)
    def call(self, logits_b):
        b = logits_b + sample_gumbel(K.shape(logits_b))
        return keras.activations.sigmoid( b/self.tau)



def my_KL_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return - K.sum(y_true*K.log(y_pred), axis=-1) 

def my_binary_KL_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    compl_y_pred = 1.0 - y_pred
    compl_y_pred = K.clip(compl_y_pred, K.epsilon(), 1)
    return - K.sum(y_true*K.log(y_pred) + (1-y_true)*K.log(compl_y_pred), axis=-1) 

def my_binary_KL_loss_stable(y_true, y_pred):

    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logits = K.log(y_pred) - K.log(1-y_pred) # sigmoid inverse
    neg_abs_logits = -K.abs(logits)
    relu_logits    = K.relu(logits)
    loss_vec = relu_logits - logits*y_true + K.log(1 + K.exp(neg_abs_logits))
    return K.sum(loss_vec)

def REC_loss(x_true, x_pred):
    x_pred = K.clip(x_pred, K.epsilon(), 1)
    return - K.sum(x_true*K.log(x_pred), axis=-1) #keras.losses.categorical_crossentropy(x_true, x_pred)

def sample_gumbel(shape,eps=K.epsilon()):
    """Inverse Sample function from Gumbel(0, 1)"""
    U = K.random_uniform(shape, 0, 1)
    return K.log(U + eps)- K.log(1-U + eps)

def SSBVAE(data_dim,n_classes,Nb,units,layers_e,layers_d,opt='adam',BN=True, summ=True,tau_ann=False,lambda_=0,alpha=1.0,beta=1.0,multilabel=False):
    if tau_ann:
        tau = K.variable(1.0, name="temperature") 
    else:
        tau = K.variable(0.67, name="temperature") #o tau fijo en 0.67=2/3
    
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    if summ:
        print("pre-encoder network:")
        pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    if summ:
        print("generator network:")
        generator.summary()

    x = Input(shape=(data_dim,))

    hidden = pre_encoder(x)
    logits_b  = Dense(Nb, activation='linear', name='logits-b')(hidden) #log(B_j/1-B_j)

    if multilabel:
        supervised_layer = Dense(n_classes, activation='sigmoid',name='sup-class')(hidden)#req n_classes  
    else:
        supervised_layer = Dense(n_classes, activation='softmax',name='sup-class')(hidden)#req n_classes
     
    encoder = Model(x, logits_b)

    def sampling(logits_b):
        # logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return keras.activations.sigmoid( b/tau )

    b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
    output = generator(b_sampled)
        
    Recon_loss = REC_loss
    kl_loss = BKL_loss(logits_b)

    def SUP_BAE_loss_pointwise(y_true, y_pred):
        return Recon_loss(y_true, y_pred) + lambda_*kl_loss(y_true, y_pred)

    margin = Nb/3.0

    if multilabel:
        pred_loss = my_binary_KL_loss_stable
    else:
        pred_loss = my_KL_loss

    # def Hamming_loss(y_true, y_pred):
        
    #     #pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    #     r = tf.reduce_sum(b_sampled*b_sampled, 1)
    #     r = tf.reshape(r, [-1, 1])
    #     D = r - 2*tf.matmul(b_sampled, tf.transpose(b_sampled)) + tf.transpose(r) #BXB
     
    #     similar_mask = K.dot(y_pred, K.transpose(y_pred)) #BXB  M_ij = I(y_i = y_j)  
    #     loss_hamming = (1.0/Nb)*K.sum(similar_mask*D + (1.0-similar_mask)*K.relu(margin-D))

    #     # return beta*pred_loss(y_true, y_pred) + alpha*loss_hamming
    #     return loss_hamming


    binary_vae = Model(inputs=x, outputs=[output,supervised_layer])

    # binary_vae.compile(optimizer=opt, loss=[SUP_BAE_loss_pointwise,Hamming_loss],loss_weights=[1., 1.], metrics=[Recon_loss,kl_loss,pred_loss])
    if tau_ann:
        return binary_vae, encoder,generator ,tau
    else:
        return binary_vae, encoder,generator
        # return binary_vae, encoder,generator, [REC_loss, BKL_loss, my_binary_KL_loss_stable, Hamming_loss]
