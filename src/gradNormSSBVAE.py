import time
from turtle import shape
import tensorflow as tf
import numpy as np
from keras import backend as K
import time
import os
from tensorflow.keras import mixed_precision

from base_networks import Hamming_loss
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
shapes =  [512, 10]

# logits_b_layer = None

def sample_gumbel(shape,eps=K.epsilon()):
    """Inverse Sample function from Gumbel(0, 1)"""
    U = K.random_uniform(shape, 0, 1)
    return K.log(U + eps)- K.log(1-U + eps)

def sampling(logits_b):
        #logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return tf.keras.activations.sigmoid( b/tau )
# total_loss        = tf.Variable(0.0, trainable=False)
# L_gradnorm        = tf.Variable(0.0, trainable=False)

@tf.function
def training_on_batch(x_batch_train, y_batch_train, z_batch_train, n_tasks,
                    alpha, epoch, logits_b_layer, gradNorm, losses_per_task):
    global sampled_layer
    # global total_loss
    # global L_gradnorm
    # print("Gradient Tape")
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pred = model(x_batch_train)
        # Task losses evaluation
        losses_value = []
        weighted_losses = []
        total_loss = 0.0 # tf.Variable(0.0, trainable=False)
        count = 0
        for i in range(n_tasks):
            for j in range(losses_per_task):
                # print(losses[count])
                if i == 0:
                    if count == 0:
                        Li = tf.math.reduce_mean((losses[count])(y_batch_train, y_pred[i]))
                    else:
                        Li = tf.math.reduce_mean((losses[count])(y_pred[i])(y_batch_train, y_pred[i]))
                else:
                    if j == 1:
                        # sampled_layer = K.function([model.get_layer(index=0).input], [model.get_layer('sampled').output])
                        logits_b = logits_b_layer([x_batch_train, K.learning_phase()])[0]
                        b_sampled = sampling(logits_b)
                        # print("b_sampled", logits_b)
                        hamming_loss = Hamming_loss(y_batch_train, y_pred[i], b_sampled)
                        Li = tf.math.reduce_mean(hamming_loss)
                    else:
                        Li = tf.math.reduce_mean((losses[count])(z_batch_train, y_pred[i]))
                losses_value.append(Li)
                # print(f"Loss {count}:  ", Li)
                # weighted Losses
                w_Li = tf.multiply(ws[count], Li)
                weighted_losses.append(w_Li)
                # add total loss
                total_loss = tf.add(total_loss, w_Li)
                count+=1
        # print("Afuera")
        if gradNorm == True:
            # L0: initial task losses
            if epoch == 0:
                count = 0
                for i in range(n_tasks):
                    for j in range(losses_per_task):
                        # L0_s[count].assign(losses_value[count])
                        L0_s[count] = losses_values[count]
                        count += 1
            
            # Gi_W
            # last_shared_layer = model.get_layer('pre-encoder')
            # last_shared_layer = model.layers[2]
            last_shared_layer = model.get_layer('pre-encoder').get_layer('last_shared_layer')
            # print("Last: ",(last_shared_layer))
            Gi_norms = []
            count = 0
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    wLi = weighted_losses[count]
                    Gi_W = tape.gradient(wLi, last_shared_layer.trainable_variables)[0]
                    # print(f'Gi_W {count}: ', Gi_W)
                    # Gradient norms
                    Gi_norm = tf.norm(Gi_W, ord=2)
                    # print(f'Gi_norm {count}: ', Gi_norm)
                    Gi_norms.append(Gi_norm)
                    count += 1
            # print("FIN GI_W")
            # Average of task gradients
            count = 0
            G_avg = 0.0 # tf.Variable(0.0, trainable=False)
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    G_avg = tf.add(G_avg, Gi_norms[count])
                    count += 1
            # print("FIN ADD AVG")
            G_avg = tf.divide(G_avg, 4.0)
            # print("FIN CALCULATE AVG")
            
            # Relative Losses
            lhat = []
            lhat_avg = 0.0 #tf.Variable(0.0, trainable=False)
            count = 0
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    lhat_i = tf.divide(losses_value[count], L0_s[count])
                    lhat.append(lhat_i)
                    lhat_avg = tf.add(lhat_avg, lhat_i)
                    count += 1
            lhat_avg = tf.divide(lhat_avg, float(len(lhat)))
            
            # Relative inverse training rates
            inv_rates = []
            count = 0
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    inv_rate = tf.divide(lhat[count], lhat_avg)
                    inv_rates.append(inv_rate)
                    count += 1
            
            #  Calculating the constant target for Eq. 2 in the GradNorm paper
            a  = tf.constant(alpha)
            C = []
            count = 0
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    Ci = tf.multiply(G_avg, tf.pow(inv_rates[count], a))
                    Ci = tf.stop_gradient(tf.identity(Ci))
                    C.append(Ci)
                    count += 1

            L_gradnorm = 0.0 # tf.Variable(0.0, trainable=False)
            count = 0
            for i in range(n_tasks):
                for j in range(losses_per_task):
                    L_gradnorm = tf.add(L_gradnorm, tf.abs(tf.subtract(Gi_norms[count], C[count])))
                    count += 1


    # Compute standard gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    # print(grads)
    # print(model.trainable_variables)
    # Model step optimization
    optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
    
    if gradNorm == True:
        # Weights step optimization
        # gradsw = tape.gradient(L_gradnorm, ws)
        # optimizer_weights.apply_gradients(zip(gradsw, ws))
        loss_step = optimizer_weights.minimize(L_gradnorm, ws, tape=tape)
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
def GradNormSSBVAE(model_to_train, X_train, Y_train, n_tasks, weights, trainable, losses_p, metrics_p, 
             epochs = 10, batch_size=512, LR=1e-2, alpha=0.12, gradNorm=True, verbose=False):

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
    # global logits_b_layer
    # global total_loss
    # global L_gradnorm

    losses = losses_p
    losses_value = []
    model = model_to_train

    logits_b_layer = K.function([model.get_layer(index=0).input, K.learning_phase()], [model.get_layer('logits-b').output])
    
    losses_values   = [ [] for _ in range(len(weights)) ]
    weights_values  = [ [] for _ in range(len(weights)) ]
    metrics = metrics_p

    ws = []     # Task weights
    L0_s = []   # Initial Losses
    for i in range(len(weights)):
        ws.append(tf.Variable(weights[i], trainable=True, constraint=tf.keras.constraints.NonNeg()))
        # L0_s.append(tf.Variable(-1.0, trainable=False))
        L0_s.append(-1.0)
    
    # Optimizers
    optimizer_model   = tf.keras.optimizers.Adam(learning_rate=LR)
    optimizer_weights = tf.keras.optimizers.Adam(learning_rate=LR*1e-2)
    # Dataset
    d = tf.data.Dataset.from_tensor_slices((X_train, Y_train[0], Y_train[1]))
    print(d)
    d.range(4)
    d.prefetch(1)#(tf.data.AUTOTUNE)
    train_dataset = d.shuffle(buffer_size = 1024, reshuffle_each_iteration=False).batch(batch_size, drop_remainder=True)
    for epoch in range(epochs):
        print(f'Start epoch {epoch}')
        # Metrics
        # for m in metrics:
        #     m.reset_state()
        # for step, (x_batch_train, y_batch_train, z_batch_train) in enumerate(train_dataset):
        for x_batch_train, y_batch_train, z_batch_train in train_dataset:
            # Standard forward pass
            print("Llamada de training on batch")
            losses_value = training_on_batch(x_batch_train, y_batch_train, z_batch_train, n_tasks, alpha, epoch, logits_b_layer, True, 2)
            # Track progress
            for i in range(len(weights)):
                metrics[i].update_state(losses_value[i])
                losses_values[i].append(losses_value[i].numpy())
                weights_values[i].append(ws[i].numpy())
        if gradNorm:
            print("Algo2")
            # Renormalizing the losses weights
            coef = 0.0
            for i in range(len(weights)):
                coef += ws[i]
            coef = tf.divide(float(len(weights)), coef)
            for i in range(len(weights)):
                ws[i].assign(tf.multiply(coef, ws[i]))
            
        # Tracking progress
        if (verbose):
            print("Epoch {:5d}:".format(epoch), end=' -> ')
            print ("weights: ", len(weights))
            print("Algo")
            for i in range(len(weights)):
                # print("Loss task {:02d}: {:.3f}".format(i, metrics[i]), end=", ")
                print("w{:02d}: {:.6f}".format(i, ws[i].numpy()), end=" ")
            print("\n", end="")

    return losses_values, weights_values