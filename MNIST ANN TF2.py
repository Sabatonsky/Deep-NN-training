# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:29:19 2023

@author: Bannikov Maxim
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
print('x_train.shape:', x_train.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

r = model.fit(x_train, y_train, batch_size=500, validation_data=(x_test, y_test), epochs=5)

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.GaussianNoise(0.2),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.GaussianNoise(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])

model2.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

f = model2.fit(x_train, y_train, batch_size=500, validation_data=(x_test, y_test), epochs=5)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history["val_loss"], label='val_loss')
plt.plot(f.history['loss'], label='loss')
plt.plot(f.history["val_loss"], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='loss')
plt.plot(r.history["val_accuracy"], label='val_loss')
plt.plot(f.history['accuracy'], label='loss')
plt.plot(f.history["val_accuracy"], label='val_loss')
plt.legend()
plt.show()

print(model.evaluate(x_test, y_test))
p_test = np.argmax(model.predict(x_test), axis = 1)

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title('True label: %s Predicted: %s' %(y_test[i], p_test[i]))
plt.show()
