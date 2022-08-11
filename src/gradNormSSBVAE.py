import time
from turtle import shape
from sklearn.preprocessing import scale
from sympy import re
import tensorflow as tf
import numpy as np
from keras import backend as K
import time
import os
# from ssb_vae import REC_loss, my_KL_loss, my_binary_KL_loss_stable

# from base_networks import BKL_loss, Hamming_loss
# mixed_precision.set_global_policy('mixed_float16')v


def Hamming_loss(y_true, y_pred, b_sampled, Nb=16):
        
    #pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    r = tf.reduce_sum(b_sampled*b_sampled, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.linalg.matmul(b_sampled, tf.transpose(b_sampled)) + tf.transpose(r) #BXB
    
    similar_mask = K.dot(y_pred, K.transpose(y_pred)) #BXB  M_ij = I(y_i = y_j)  
    loss_hamming = (1.0/Nb)*K.sum(similar_mask*D + (1.0-similar_mask)*K.relu((Nb/3.0)-D))
    # loss_hamming = (1.0/Nb)*K.mean(similar_mask*D + (1.0-similar_mask)*K.relu((Nb/3.0)-D))


    # return beta*pred_loss(y_true, y_pred) + alpha*loss_hamming
    return loss_hamming

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
    # return - K.sum(x_true*K.log(x_pred), axis=-1) 
    return tf.keras.losses.categorical_crossentropy(x_true, x_pred)

def BKL_loss(logits_b):
    p_b = tf.keras.activations.sigmoid(logits_b) #B_j = Q(b_j) probability of b_j
    Nb = K.int_shape(p_b)[1]
    ep = K.epsilon()
    def KL(y_true, y_pred):
        return (Nb*np.log(2) + K.sum( p_b*K.log(p_b + ep) + (1-p_b)* K.log(1-p_b +ep),axis=1))
    return KL


# Global vars
losses_values     = None
weights_values    = None
metrics           = None
ws                = None
L0_s              = None
optimizer_model   = None
optimizer_weights = None
model             = None
losses            = None
logits_b_layer    = None

@tf.function
def training_on_batch(x_batch_train, y_batch_train, z_batch_train, alpha, gamma, epoch, gradNorm):
    sampled_layer = model.get_layer("sampled")
    # logits_b_layer = tf.keras.models.Model(
    #     inputs=model.input,
    #     outputs=model.get_layer("logits-b").output
    # )

    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pred = model(x_batch_train)
        # Task losses evaluation
        losses_value = []
        weighted_losses = []
        total_loss = 0.0
        
        logits_b = logits_b_layer(x_batch_train)
        b_sampled = sampled_layer(logits_b)

        rec_loss = tf.reduce_mean(REC_loss(y_batch_train, y_pred[0]))
        bkl_loss = tf.reduce_mean(BKL_loss(logits_b)(y_batch_train, y_pred[0]))
        pred_loss = tf.reduce_mean(my_KL_loss(z_batch_train, y_pred[1]))
        
        hamming_loss = Hamming_loss(z_batch_train, y_pred[1], b_sampled)
        
        # losses_value.append(rec_loss) 
        losses_value.append(bkl_loss)
        # losses_value.append(rec_loss+bkl_loss)
        # losses_value.append(pred_loss+hamming_loss)
        losses_value.append(pred_loss)
        losses_value.append(hamming_loss)

        # rec_loss + w0*bkl_loss + w1*pred_loss + w2*hloss

        for i in range(len(losses_value)):
            wLi = tf.multiply(ws[i], losses_value[i])
            weighted_losses.append(wLi)
            total_loss = tf.add(total_loss, wLi)
            # total_loss += wLi
        total_loss = tf.add(total_loss, rec_loss)
        # total_loss = tf.add(total_loss, 0.4*hamming_loss)
        # total_loss += rec_loss

        if gradNorm:
            # L0: initial task losses
            if epoch == 0:
                for i in range(len(ws)):
                    L0_s[i].assign(losses_value[i])
            
            # Gi_W
            last_shared_layer = model.get_layer('pre-encoder').get_layer('last_shared_layer')
            Gi_norms = []
            for i in range(len(ws)):
                wLi = weighted_losses[i]
                Gi_W = tape.gradient(wLi, last_shared_layer.trainable_variables)[0]
                # Gradient norms
                Gi_norm = tf.norm(Gi_W, ord=2)
                Gi_norms.append(Gi_norm)
            # Average of task gradients
            G_avg = 0.0
            for i in range(len(ws)):
                G_avg = tf.add(G_avg, Gi_norms[i])
                # G_avg += Gi_norms[i]
            G_avg = tf.divide(G_avg, float(len(ws)))
            # G_avg /= tf.cast(len(ws), dtype=tf.float64)
            
            # Relative Losses
            lhat = []
            lhat_avg = 0.0 
            for i in range(len(ws)):
                lhat_i = tf.divide(losses_value[i], L0_s[i])
                lhat.append(lhat_i)
                lhat_avg = tf.add(lhat_avg, lhat_i)
                # lhat_avg += lhat_i
                
            lhat_avg = tf.divide(lhat_avg, float(len(ws)))
            
            # Relative inverse training rates
            inv_rates = []
            for i in range(len(ws)):
                inv_rate = tf.divide(lhat[i], lhat_avg)
                inv_rates.append(inv_rate)
            
            #  Calculating the constant target for Eq. 2 in the GradNorm paper
            a  = tf.constant(alpha)
            C = []
            for i in range(len(ws)):
                Ci = tf.multiply(G_avg, tf.pow(inv_rates[i], tf.cast(a, dtype=tf.float32))) 
                Ci = tf.stop_gradient(tf.identity(Ci)) #Make constant the term
                C.append(Ci)

            L_gradnorm = 0.0
            for i in range(len(ws)):
                # L_gradnorm += tf.norm(Gi_norms[i]-C[i], ord=1)
                    L_gradnorm = tf.add(L_gradnorm,
                                        tf.norm(
                                        tf.subtract(Gi_norms[i], C[i])
                                        ,ord=1)
                    )


    # Compute standard gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    # Model step optimization
    optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
    
    if gradNorm:
        # Weights step optimization
        gradsw = tape.gradient(L_gradnorm, ws)
        # gradsw = list(map(lambda x: tf.multiply(x,-1), gradsw))
        # optimizer_weights.apply_gradients(zip(negative_gradient, ws))
        optimizer_weights.apply_gradients(zip(gradsw, ws))
        # loss_step = optimizer_weights.minimize(L_gradnorm, ws, tape=tape)

    return losses_value
    
'''
Parameters:
* model: keras model to train
* X_train, Y_train: input and output to train
* n_tasks: Number of tasks of model
* weights: Inital weights of tasks
* losses: losses objects to each task
* metrics: metrics to evaluate per task
* alpha: hyper parameter of gradNorm algorithm
* gamma: gamma parameter of SSB-VAE
* verbose: print status of trainig
'''    
def GradNormSSBVAE(model_to_train, X_train, Y_train, weights, 
             epochs = 10, batch_size=512, LR=1e-2, alpha=0.12, gamma=1.0, gradNorm=True, verbose=True):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Build in CUDA: ", tf.test.is_built_with_cuda())
    global losses_values
    global weights_values
    global ws
    global L0_s
    global optimizer_model
    global optimizer_weights
    global losses
    global model
    global logits_b_layer

    losses_value = []
    model = model_to_train

    logits_b_layer = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("logits-b").output
    )
    
    losses_values   = [ [] for _ in range(len(weights)) ]
    weights_values  = [ [] for _ in range(len(weights)) ]

    ws   = []   # Task weights
    L0_s = []   # Initial Losses
    for i in range(len(weights)):
        ws.append(tf.Variable(weights[i], trainable=True, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32))
        L0_s.append(tf.Variable(-1.0, trainable=False, dtype=tf.float32))
    # Optimizers
    lr_schedule_model = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=3000,
        decay_rate=0.1)
    lr_schedule_ws = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=3000,
        decay_rate=0.1)
    # optimizer_model   = tf.keras.optimizers.Adam(learning_rate=lr_schedule_model)
    # optimizer_weights = tf.keras.optimizers.Adam(learning_rate=lr_schedule_ws)
    # optimizer_model   = tf.keras.optimizers.Adam()
    # optimizer_weights = tf.keras.optimizers.Adam()
    optimizer_model   = tf.keras.optimizers.Adam(learning_rate=LR)
    optimizer_weights = tf.keras.optimizers.Adam(learning_rate=LR)
    # Dataset
    d = tf.data.Dataset.from_tensor_slices((X_train, Y_train[0], Y_train[1]))
    d.range(4)
    d.prefetch(1)#(tf.data.AUTOTUNE)
    train_dataset = d.shuffle(buffer_size = 1024, reshuffle_each_iteration=False).batch(batch_size, drop_remainder=True)
    for epoch in range(epochs):
        # Metrics
        # for m in metrics:
        #     m.reset_state()
        for x_batch_train, y_batch_train, z_batch_train in train_dataset:
            # print(y_batch_train)
            # Standard forward pass
            losses_value = training_on_batch(x_batch_train, y_batch_train, z_batch_train, alpha, gamma, epoch, gradNorm)
            # Track progress
            for i in range(len(weights)):
                losses_values[i].append(losses_value[i].numpy())
                weights_values[i].append(ws[i].numpy())
        if gradNorm:
            # Renormalizing the losses weights
            coef = 0.0
            for i in range(len(ws)):
                coef += ws[i]
            coef = tf.divide(float(len(ws)), coef)
            # coef = tf.divide(3.6, coef)
            for i in range(len(ws)):
                ws[i].assign(tf.multiply(coef, ws[i]))
            
        # Tracking progress
        if (verbose):
            print("Epoch {:5d}:".format(epoch), end=' -> ')
            for i in range(len(losses_value)):
                print("Loss task {:02d}: {:.3f}".format(i, losses_value[i]), end=", ")
                print("w{:02d}: {:.10f}".format(i, ws[i].numpy()), end=" ")
            print("\n", end="")
    for w in ws:
        del(w)
    for l0 in L0_s:
        del(l0)
    del(logits_b_layer)

    return losses_values, weights_values