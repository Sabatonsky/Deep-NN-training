# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 22:06:16 2023

@author: Bannikov Maxim
"""

import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
from keras.layers import Dense

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train - x_train.mean()) / x_train.std()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = (x_test - x_test.mean()) / x_test.std()
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
print('x_train.shape:', x_train.shape)

M = 20
nHidden = 2
log_lr = -4
log_l2 = -2
max_tries = 30

best_validation_rate = 0
best_hls = None
best_lr = None
best_l2 = None

for _ in range (max_tries):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(784)))
    for i in range(nHidden):
        model.add(Dense(M,
                        activation='relu',
                        kernel_regularizer=regularizers.L2(l2=10**log_l2),
                        bias_regularizer=regularizers.L2(l2=10**log_l2)))
    model.add(Dense(10, activation = 'softmax'))
    print(model.summary())
    opt = tf.keras.optimizers.Adam(10**log_lr)
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    r = model.fit(x_train, y_train, batch_size=500, validation_data=(x_test, y_test), epochs=10)
    
    validation_accuracy = r.history['val_accuracy'][-1]
    train_accuracy = r.history['accuracy'][-1]
    print('val accuracy', validation_accuracy, 'train accuracy', train_accuracy, [M]*nHidden, log_lr, log_l2)
    
    if validation_accuracy > best_validation_rate:
        best_validation_rate = validation_accuracy
        best_M = M
        best_nHidden = nHidden
        best_lr = log_lr
        best_l2 = log_l2
        
    nHidden = best_nHidden + np.random.randint(-1, 2)
    nHidden = max(1, nHidden)
    M = best_M + np.random.randint(-1, 2)*10
    log_lr = best_lr + np.random.randint(-1, 2)
    log_l2 = best_l2 + np.random.randint(-1, 2)
    
print("best accuracy", best_validation_rate)
print('best settings:')
print("best_M:", best_M)
print("best_nHidden", best_nHidden)
print("best_lr:", best_lr)
print("best_l2", best_l2)
