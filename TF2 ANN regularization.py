# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:29:19 2023

@author: Bannikov Maxim
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers

N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

plt.scatter(X[:,0], X[:,1], c=Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,
                          input_shape=(2,), 
                          activation='relu',
                          kernel_regularizer=regularizers.L2(l2=1e-4),
                          bias_regularizer=regularizers.L2(l2=1e-4)),
    tf.keras.layers.Dense(1)
    ])

opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mse')
r = model.fit(X, Y, batch_size = 100, epochs = 100)

plt.plot(r.history['loss'])
P = model.predict(X).flatten()

line = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, color='red')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()
 