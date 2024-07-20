# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:13:20 2023

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def feedforward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z
    
def softmax(Y):
    T = np.zeros((len(Y), max(Y) + 1))
    for i in range(len(Y)):
        T[i, Y[i]] = 1
    return T

def cost(T, Y):
    tot = T*np.log(Y)
    return tot.sum()
    
def classification_rate(P, Y):
    n_correct = 0
    n_total = 0
    for i in range(Y.shape[0]):
        if Y[i] == P[i]:
            n_correct += 1
        n_total += 1
    return float(n_correct) / n_total

dir_path = os.path.dirname(r"C:\Users\AMD\Desktop\Training_code")

train_data = pd.read_csv("mnist_train.csv").to_numpy(dtype=np.float32)
Xtrain = train_data[:,1:]/255
Ytrain = train_data[:,0].astype(np.int32)
Ttrain = softmax(Ytrain)
test_data = pd.read_csv("mnist_test.csv").to_numpy(dtype=np.float32)
Xtest = test_data[:,1:]/255
Ytest = test_data[:,0].astype(np.int32)
Ttest = softmax(Ytest)

D = Xtrain.shape[1]
K = 10
M = 25

W1 = np.random.randn(D,M) / np.sqrt(D)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

learning_rate = 10e-6
Costs_train = []
Costs_test = []

for epoch in range(10001):
    Out_train, hidden_train = feedforward(Xtrain, W1, b1, W2, b2)
    Out_test, hidden_test = feedforward(Xtest, W1, b1, W2, b2)
    
    Ptrain = np.argmax(Out_train, axis = 1)
    Ptest = np.argmax(Out_test, axis = 1)
    
    Cost_train = cost(Ttrain, Out_train)
    Cost_test = cost(Ttest, Out_test)
    Costs_train.append(Cost_train)
    Costs_test.append(Cost_test)
    
    Rate_train = classification_rate(Ptrain, Ytrain)
    Rate_test = classification_rate(Ptest, Ytest)
    
    W2 += learning_rate*hidden_train.T.dot(Ttrain - Out_train)
    b2 += learning_rate*(Ttrain - Out_train).sum(axis = 0)
    W1 += learning_rate*Xtrain.T.dot((Ttrain - Out_train).dot(W2.T)*(1-hidden_train**2))
    b1 += learning_rate*((Ttrain - Out_train).dot(W2.T)*(hidden_train)*(1-hidden_train)).sum(axis = 0)
    
    if epoch % 100 == 0:
        print("Epoch number:", epoch)
        print("Train loss:", Cost_train, "Train rate:", round(Rate_train,2))
        print("Test loss:", Cost_test, "Test rate:", round(Rate_test,2))
        
plt.plot(Costs_train)
plt.plot(Costs_test)
plt.legend(['Train', 'Test'])
plt.show()
