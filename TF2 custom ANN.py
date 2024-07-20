# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 20:03:28 2023

@author: Bannikov Maxim
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs):
        super(LinearRegression, self).__init__()
        self.W = tf.Variable(
            tf.random_normal_initializer()((num_inputs, num_outputs)))
        self.b = tf.Variable(tf.zeros(num_outputs))
        self.params = [self.W, self.b]
        
    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b
    
N = 100
D = 1
K = 1
X = np.random.random((N, D)) * 2 - 1
w = np.random.randn(D, K)
b = np.random.randn()
Y = X.dot(w) + b + np.random.randn(N, 1)* 0.1

plt.scatter(X, Y)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

def get_loss(model, inputs, targets):
    predictions = model(inputs)
    error = targets - predictions
    return tf.reduce_mean(tf.square(error))

def get_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = get_loss(model, inputs, targets)
    return tape.gradient(loss_value, model.params)

model = LinearRegression(D, K)

print(model.W, model.b)
losses = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

for i in range(100):
    grads = get_grad(model, X, Y)
    optimizer.apply_gradients(zip(grads, model.params))
    loss = get_loss(model, X, Y)
    losses.append(loss)
    
plt.plot(losses)
