# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:16:57 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

def forward(X, b1, W1, b2, W2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1)) # Функция sigmoid с линейной регрессией внутри
    A = Z.dot(W2) + b2 # Линейная регрессия с сигмоидной функцией Z.
    expA = np.exp(A) 
    Y = expA / expA.sum(axis = 1, keepdims = True) # Softmax
    return Y, Z

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        if Y[i] == P[i]:
            n_correct += 1
        n_total += 1
    return float(n_correct) / n_total

def cat_to_ind(Yin):
    N = len(Yin)
    K = max(Yin) + 1
    Yout = np.zeros((N, K))
    for n in range(N):
        Yout[n, Yin[n]] = 1
    return Yout

def cost(T, Y):
    tot = T * np.log(Y) # Предыдущая функция loss была T * np.log(Y) + (1 - T) * np.log(1 - Y), где Y и T у нас 1 или 0
    # Однако шутки по бернульски закончились. Теперь loss матричный. Categorical Cross-Entropy, вся хуйня.
    return tot.sum()

def derivative_w2(Z, T, Y):
    return Z.T.dot(T - Y)

def derivative_b2(T, Y):
    return (T - Y).sum(axis = 0)

def derivative_w1(X, Z, T, Y, W2):
    return X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z)) 
    
def derivative_b1(Z, T, Y, W2):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis = 0)

N_class = 500
D = 2
M = 3
K = 3
X1 = np.random.randn(N_class, D) + np.array([0, -2])
X2 = np.random.randn(N_class, D) + np.array([2, 2])
X3 = np.random.randn(N_class, D) + np.array([-2, 2])

X = np.vstack([X1,X2,X3])
Y = np.array([0]*N_class + [1]*N_class + [2]*N_class)
N = len(Y)
T = np.zeros([N,K])
for i in range(N):
    T[i,Y[i]] = 1
    
W1 = np.random.rand(D, M)
b1 = np.random.rand(M)
W2 = np.random.rand(M, K)
b2 = np.random.rand(K)

learning_rate = 10e-7
costs = []

for epoch in range(100001):
    output, hidden = forward(X, b1, W1, b2, W2)
    if epoch % 10000 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis = 1) #predictions array for calculation of classification rate
        r = classification_rate(Y, P)
        costs.append(c)
        print("Epoch number:", epoch, "loss:", c, "classification rate:", round(r,2))
        plt.scatter(X[:,0], X[:,1], c = output)
        plt.show()

    W2 += learning_rate * derivative_w2(hidden, T, output)
    b2 += learning_rate * derivative_b2(T, output)
    W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
    b1 += learning_rate * derivative_b1(hidden, T, output, W2)
 
