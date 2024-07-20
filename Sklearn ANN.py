# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:53:46 2023

@author: Bannikov Maxim
"""

from utils import get_data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

X, Y = get_data()

X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

model = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)
model.fit(Xtrain, Ytrain)

train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)

plt.plot(model.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)
