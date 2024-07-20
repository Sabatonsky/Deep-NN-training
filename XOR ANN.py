# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:29:08 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2
K = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ])

T = np.array([[0],[1],[1],[0]])

M = 10
W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,1)
b2 = np.random.randn(1)

def sigmoid(X, W, b):
    return 1 / (1 + np.exp(-X.dot(W) - b))

def feedforward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    Y = sigmoid(Z, W2, b2)
    return Y, Z

def cost(T, Y):
    return -np.mean(T * np.log(Y) + (1-T) * np.log(1-Y))

def classification_rate(T, Y):
    return np.mean(T == Y)

costs = []
lr = 10e-5

for epoch in range(100001):
    Y, Z = feedforward(X, W1, b1, W2, b2)
    if epoch % 10000 == 0:
        c = cost(T, Y)
        P = np.round(Y, 0)
        r = classification_rate(T, P)
        print("Epoch number:", epoch, "loss:", c, "classification rate:", round(r,2))
        plt.scatter(X[:,0], X[:,1], c = P)
        plt.show()
        
    W2 += lr * Z.T.dot(T - Y)
    b2 += lr * (T - Y).sum(axis = 0)
    dZ = (T - Y).dot(W2.T) * (1 - Z**2)
    W1 += lr * X.T.dot(dZ)
    b1 += lr * (dZ).sum(axis = 0)






