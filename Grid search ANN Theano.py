# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 09:13:44 2023

@author: 79296
"""

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import utils as ut
from theano import ANN

X, Y = ut.get_spiral()
X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
plt.scatter(X[:,0], X[:,1], c=Y)

hidden_layer_sizes = [[300], [100, 100], [50, 50, 50]]
learning_rates = [1e-4, 1e-3, 1e-2]
l2_penalties = [0., 0.1, 1.0]

best_validation_rate = 0
best_hls = None
best_lr = None
best_l2 = None
for hls in hidden_layer_sizes:
    for lr in learning_rates:
        for l2 in l2_penalties:
            model = ANN(hls)
            model.fit(Xtrain, Ytrain, learning_rate=lr, reg=l2, mu=0.99, epoch=3000, show_accuracy = True)
            validation_accuracy = model.score(Xtest, Ytest)
            train_accuracy = model.score(Xtrain, Ytrain)
            