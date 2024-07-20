# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:54:04 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from utils import get_data

def y2indicator(y):
    N = len(y)
    K = max(y) + 1
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z

def classification_rate(Y,P):
    return np.mean(Y == P)

def cost(Y, T):
    return -np.mean(T * np.log(Y))


X, Tcat = get_data()
X, Tcat = shuffle(X, Tcat)
Tcat = Tcat.astype(np.int32)
K = len(set(Tcat))

X_train = X[:-100,:]
Y_train = Tcat[:-100]

X_test = X[-100:,:]
Y_test = Tcat[-100:]

T_train = y2indicator(Y_train)
T_test = y2indicator(Y_test)

N_train = len(Y_train)
N_test = len(Y_test)
D = X.shape[1]
M = 10

W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)
learning_rate = 10e-7

costs = []

for epoch in range(100001):
    Y, Z = forward(X_train, W1, b1, W2, b2)
    c = cost(Y, T_train)
    costs.append(c)
    if epoch % 1000 == 0:
        P = np.argmax(Y, axis = 1)
        r = classification_rate(P,Y_train)
        print("Epoch number:", epoch, "Cost:", c, "Classification rate:", r)
    W2 += learning_rate * Z.T.dot(T_train - Y)
    b2 += learning_rate * (T_train - Y).sum(axis = 0)
    dZ = (T_train - Y).dot(W2.T) * (1 - Z**2)
    W1 += learning_rate * X_train.T.dot(dZ)
    b1 += learning_rate * dZ.sum(axis = 0)

Y, Z = forward(X_test, W1, b1, W2, b2)
c = cost(Y, T_test)
P = np.argmax(Y, axis = 1)
r = classification_rate(P,Y_test)
print("Test-set cost:", c, "Test-set classification rate:", r)
 
plt.plot(costs)
