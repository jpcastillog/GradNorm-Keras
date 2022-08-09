import time
from turtle import shape
import tensorflow as tf
import numpy as np
from keras import backend as K
import time
import os
from ssb_vae import REC_loss, my_KL_loss, my_binary_KL_loss_stable

from base_networks import BKL_loss, Hamming_loss
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
trainable         = None
logits_b_layer    = None

@tf.function
def training_on_batch(x_batch_train, y_batch_train, z_batch_train, alpha, gamma, epoch, gradNorm):
    sampled_layer = model.get_layer("sampled")
    logits_b_layer = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("logits-b").output
    )

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
        losses_value.append(gamma*(pred_loss)+hamming_loss)
        # losses_value.append(pred_loss)
        # losses_value.append(hamming_loss)

        for i in range(len(losses_value)):
            wLi = tf.multiply(ws[i], losses_value[i])
            weighted_losses.append(wLi)
            total_loss = tf.add(total_loss, wLi)
        total_loss = tf.add(total_loss, rec_loss)

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
            G_avg = tf.divide(G_avg, float(len(ws)))
            
            # Relative Losses
            lhat = []
            lhat_avg = 0.0 
            for i in range(len(ws)):
                lhat_i = tf.divide(losses_value[i], L0_s[i])
                lhat.append(lhat_i)
                lhat_avg = tf.add(lhat_avg, lhat_i)
                
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
                Ci = tf.multiply(G_avg, tf.pow(inv_rates[i], a)) 
                Ci = tf.stop_gradient(tf.identity(Ci)) #Make constant the term
                C.append(Ci)

            L_gradnorm = 0.0
            for i in range(len(ws)):
                    L_gradnorm = tf.add(L_gradnorm,
                                        tf.norm(
                                        tf.abs(tf.subtract(Gi_norms[i], C[i]))
                                        ,ord=1)
                    )


    # Compute standard gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    # Model step optimization
    optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
    
    if gradNorm:
        # Weights step optimization
        gradsw = tape.gradient(L_gradnorm, ws)
        negative_gradient = list(map(lambda x: tf.multiply(x,-1), gradsw))
        optimizer_weights.apply_gradients(zip(negative_gradient, ws))
        # optimizer_weights.apply_gradients(zip(gradsw, ws))
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
def GradNormSSBVAE(model_to_train, X_train, Y_train, weights, losses_p, metrics_p, 
             epochs = 10, batch_size=512, LR=1e-2, alpha=0.12, gamma=1.0, gradNorm=True, verbose=False):

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
    global logits_b_layer
    global trainable

    losses = losses_p
    losses_value = []
    model = model_to_train

    logits_b_layer = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("logits-b").output
    )
    
    losses_values   = [ [] for _ in range(len(weights)) ]
    weights_values  = [ [] for _ in range(len(weights)) ]
    metrics = metrics_p

    ws   = []   # Task weights
    L0_s = []   # Initial Losses
    print("len(weights): ", len(weights))
    for i in range(len(weights)):
        ws.append(tf.Variable(weights[i], trainable=True, constraint=tf.keras.constraints.NonNeg()))
        L0_s.append(tf.Variable(-1.0, trainable=False))
    # Optimizers
    lr_schedule_model = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=3000,
        decay_rate=0.1)
    lr_schedule_ws = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=3000,
        decay_rate=0.1)
    optimizer_model   = tf.keras.optimizers.Adam(learning_rate=lr_schedule_model)
    optimizer_weights = tf.keras.optimizers.Adam(learning_rate=lr_schedule_ws)
    # optimizer_model   = tf.keras.optimizers.Adam(learning_rate=LR)
    # optimizer_weights = tf.keras.optimizers.Adam(learning_rate=LR)
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
            for i in range(len(ws)):
                ws[i].assign(tf.multiply(coef, ws[i]))
            
        # Tracking progress
        if (verbose):
            print("Epoch {:5d}:".format(epoch), end=' -> ')
            for i in range(len(losses_value)):
                print("Loss task {:02d}: {:.3f}".format(i, losses_value[i]), end=", ")
                print("w{:02d}: {:.10f}".format(i, ws[i].numpy()), end=" ")
            print("\n", end="")

    return losses_values, weights_values