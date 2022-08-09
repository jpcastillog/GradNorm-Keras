# Implementation use Tensorflow 2
import time
from turtle import shape
import tensorflow as tf
import numpy as np
from keras import backend as K
import time
import os
from tensorflow.keras import mixed_precision

from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')v

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


# @tf.function
def training_on_batch(x_batch_train, y_batch_train, n_tasks,
                    alpha, epoch, gradNorm):
    # global total_loss
    # global L_gradnorm
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pred = model(x_batch_train, training=True)
        # Task losses evaluation
        losses_value = []
        weighted_losses = []
        weighted_scaled_losses = []
        total_loss = 0.0 # tf.Variable(0.0, trainable=False)
        total_scaled_loss = 0.0
        for i in range(n_tasks):
            Li = (losses[i])(y_true=y_batch_train[:,i], y_pred=y_pred[i])
            losses_value.append(Li)
            # weighted Losses
            w_Li = tf.multiply(ws[i], Li)
            # print("Li: ", w_Li.numpy())
            weighted_losses.append(w_Li)
            # add total loss
            total_loss = tf.add(total_loss, w_Li)
            
        if gradNorm == True:
            # L0: initial task losses
            if epoch == 0:
                for i in range(n_tasks):
                    L0_s[i].assign(losses_value[i])
            
            # Gi_W
            last_shared_layer = model.get_layer('last_shared_layer')
            Gi_norms = []
            for i in range(n_tasks):
                wLi = weighted_losses[i]
                Gi_W = tape.gradient(wLi, last_shared_layer.trainable_variables)[0]
                # Gradient norms
                Gi_norm = tf.norm(Gi_W, ord=2)
                Gi_norms.append(Gi_norm)
            
            # Average of task gradients
            G_avg = 0.0 # tf.Variable(0.0, trainable=False)
            for i in range(n_tasks):
                G_avg = tf.add(G_avg, Gi_norms[i])
            G_avg = tf.divide(G_avg, float(n_tasks))
            
            # Relative Losses
            lhat = []
            lhat_avg = 0.0 #tf.Variable(0.0, trainable=False)
            for i in range(n_tasks):
                lhat_i = tf.divide(losses_value[i], L0_s[i])
                lhat.append(lhat_i)
                lhat_avg = tf.add(lhat_avg, lhat_i)
            lhat_avg = tf.divide(lhat_avg, float(n_tasks))
            
            # Relative inverse training rates
            inv_rates = []
            for i in range(n_tasks):
                inv_rate = tf.divide(lhat[i], lhat_avg)
                inv_rates.append(inv_rate)
            
            #  Calculating the constant target for Eq. 2 in the GradNorm paper
            a  = tf.constant(alpha)
            C = []
            for i in range(n_tasks):
                Ci = tf.multiply(G_avg, tf.pow(inv_rates[i], a))
                Ci = tf.stop_gradient(tf.identity(Ci))
                C.append(Ci)

            L_gradnorm = 0.0 # tf.Variable(0.0, trainable=False)
            for i in range(n_tasks):
                L_gradnorm = tf.add(L_gradnorm, tf.norm(tf.abs(tf.subtract(Gi_norms[i], C[i])), ord=1))


    # Compute standard gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    # Model step optimization
    optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
    
    if gradNorm == True:
        # Weights step optimization
        gradsw = tape.gradient(L_gradnorm, ws)
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
* verbose: print status of trainig
'''    
def GradNorm(model_to_train, X_train, Y_train, n_tasks, weights, losses_p, metrics_p,
             epochs = 10, batch_size=128, LR=[1e-2, 1e-2], alpha=0.12, gradNorm=True, verbose=False):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Build in CUDA: ", tf.test.is_built_with_cuda())
    global losses_values
    global weights_values
    global metrics
    global ws
    global L0_s
    global optimizer_model
    global optimizer_weights
    global losses
    global model

    losses = losses_p
    losses_value = []
    model = model_to_train
    
    losses_values   = [ [] for _ in range(n_tasks) ]
    weights_values  = [ [] for _ in range(n_tasks) ]
    metrics = metrics_p

    if (len(weights) != n_tasks) or (len(losses) != n_tasks):
        raise Exception('Number of losses and weights need to be equal of tasks number')
    ws = []     # Task weights
    L0_s = []   # Initial Losses
    for i in range(n_tasks):
        ws.append(tf.Variable(1.0, trainable=True, constraint=tf.keras.constraints.NonNeg()))
        L0_s.append(tf.Variable(-1.0, trainable=False))
    
    # Optimizers
    optimizer_model   = tf.keras.optimizers.Adam(learning_rate=LR[0])
    optimizer_weights = tf.keras.optimizers.Adam(learning_rate=LR[1])
    # Dataset
    d = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    d.range(4)
    d.prefetch(1)
    train_dataset = d.shuffle(buffer_size = 1024, reshuffle_each_iteration=False).batch(batch_size, drop_remainder=True)
    for epoch in range(epochs):
        # print(f'Start epoch {epoch}')
        # Metrics
        # for m in metrics:
        #     m.reset_state()
        
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Standard forward pass
            losses_value = training_on_batch(x_batch_train, y_batch_train, n_tasks, alpha, epoch, gradNorm)
            # Track progress
            for i in range(n_tasks):
                metrics[i].update_state(losses_value[i])
                losses_values[i].append(losses_value[i].numpy())
                weights_values[i].append(ws[i].numpy())
        if gradNorm:
            # Renormalizing the losses weights
            coef = 0.0
            for i in range(n_tasks):
                coef += ws[i]
            coef = tf.divide(float(n_tasks), coef)
            for i in range(n_tasks):
                ws[i].assign(tf.multiply(coef, ws[i]))
            
        # Tracking progress
        if (verbose):
            print("Epoch {:5d}:".format(epoch), end=' -> ')
            for i in range(n_tasks):
                print("Loss task {:02d}: {:.3f}".format(i, metrics[i].result().numpy()), end=", ")
                print("w{:02d}: {:.6f}".format(i, ws[i].numpy()), end=" ")
            print("\n", end="")
    for w in ws:
        del(w)
    for l0 in L0_s:
        del(l0)
    del(ws)
    del(L0_s)
    return losses_values, weights_values