# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:13:20 2023

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feedforward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z
    
def y2indicator(Y):
    T = np.zeros((len(Y), max(Y) + 1))
    for i in range(len(Y)):
        T[i, Y[i]] = 1
    return T

def cost(T, Y):
    tot = T*np.log(Y)
    return -tot.mean()

def predict(P):
    return np.argmax(P, axis=1)

def error_rate(P, Y):
    prediction = predict(P)
    return np.mean(prediction != Y)

def get_batches(X, Y, batch_size):
    num_batches = int(Y.shape / batch_size)
    for j in range(num_batches):
        Xb = X[j * batch_size:(j + 1) * batch_size]
        Yb = Y[j * batch_size:(j + 1) * batch_size]
    return Xb, Yb
    
def get_normalized_data():
    train_data = pd.read_csv("mnist_train.csv").to_numpy(dtype=np.float32)
    np.random.shuffle(train_data)
    Xtrain = train_data[:,1:]
    Ytrain = train_data[:,0].astype(np.int32)
    test_data = pd.read_csv("mnist_test.csv").to_numpy(dtype=np.float32)
    np.random.shuffle(test_data)
    Xtest = test_data[:,1:]
    Ytest = test_data[:,0].astype(np.int32)
    
    mu = Xtrain.mean(axis = 0)
    std = Xtrain.std(axis = 0)
    
    idx = np.where(std == 0)[0]
    assert(np.all(std[idx] == 0))
    np.place(std, std == 0, 1)
    
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    
    return Xtrain, Xtest, Ytrain, Ytest


def gradW1(Xtrain, Ytrain_ind, Out_train, hidden_train, W2):
    return Xtrain.T.dot((Ytrain_ind - Out_train).dot(W2.T)*(1-hidden_train**2))    

def gradb1(Ytrain_ind, Out_train, hidden_train, W2):
    return ((Ytrain_ind - Out_train).dot(W2.T)*(1-hidden_train**2)).sum(axis = 0)
            
def gradW2(Ytrain_ind, Out_train, hidden_train):
    return hidden_train.T.dot(Ytrain_ind - Out_train)  

def gradb2(Ytrain_ind, Out_train):
    return (Ytrain_ind - Out_train).sum(axis = 0)

def linear_benchmark():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    K = Ytrain_ind.shape[1]
    M = 25

    W1 = np.random.randn(D,M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M,K)
    b2 = np.zeros(K)

    learning_rate = 10e-6
    reg = 0.01
    n_iters = 101
    
    Costs_train = []
    Costs_test = []
    train_class_errors = []
    test_class_errors = []

    for epoch in range(n_iters):
        Out_train, hidden_train = feedforward(Xtrain, W1, b1, W2, b2)
        Out_test, hidden_test = feedforward(Xtest, W1, b1, W2, b2)
        
        Ptrain = np.argmax(Out_train, axis = 1)
        Ptest = np.argmax(Out_test, axis = 1) 
        
        Cost_train = cost(Ytrain_ind, Out_train)
        Cost_test = cost(Ytest_ind, Out_test)
        Costs_train.append(Cost_train)
        Costs_test.append(Cost_test)
        
        Rate_train = error_rate(Ptrain, Ytrain)
        Rate_test = error_rate(Ptest, Ytest)
        train_class_errors.append(Rate_train)
        test_class_errors.append(Rate_test)
        
        W2 += learning_rate*(gradW2(Ytrain_ind, Out_train, hidden_train) - reg*W2)
        b2 += learning_rate*(gradb2(Ytrain_ind, Out_train) - reg*b2)
        W1 += learning_rate*(gradW1(Xtrain, Ytrain_ind, Out_train, hidden_train, W2) - reg*W1)
        b1 += learning_rate*(gradb1(Ytrain_ind, Out_train, hidden_train, W2) - reg*b1)
        
        if epoch % 100 == 0:
            print("Epoch number:", epoch)
            print("Train loss:", round(Cost_train,2), "Train error:", round(Rate_train, 2))
            print("Test loss:", round(Cost_test,2), "Test error:", round(Rate_test, 2))
        
    plt.plot(Costs_train)
    plt.plot(Costs_test)
    plt.legend(['Train', 'Test'])
    plt.show()

def get_data():
    df = pd.read_csv('ecommerce_data.csv', encoding = 'UTF-8')
    data = df.to_numpy()
    
    X = data[:,:-1]
    Y = data[:, -1]
    
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,1].std()
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
        
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    for n in range(N):
        t = int(X[n, D-1])
        X2[n,t+D-1] = 1
        
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)
    
    return X2, Y
    
def get_binary_data():
    
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    
    return X2, Y2
    
if __name__ == '__main__':
    linear_benchmark()
