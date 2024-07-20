# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:42:25 2023

@author: Bannikov Maxim
"""

import numpy as np
import utils as ut
import tensorflow as tf
import matplotlib.pyplot as plt

def feedforward(X, W1, b1, W2, b2, gamma, beta, eps):
    X_in = X.dot(W1) + b1
    X_out, cache, mu, var = batchnorm_forward(X_in, gamma, beta, eps)
    Z = np.tanh(X_out)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z, X_out, cache, mu, var


def batchnorm_forward(x, gamma, beta, eps):

    N, D = x.shape
    mu = 1./N * np.sum(x, axis = 0)
    xmu = x - mu
    sq = xmu ** 2
    var = 1./N * np.sum(sq, axis = 0)
    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar
    gammax = gamma * xhat
    out = gammax + beta
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    
    return out, cache, mu, var

def feedforward_test(X, W1, b1, W2, b2, gamma, beta, eps, mu, var):
    X_in = X.dot(W1) + b1
    X_out, cache = batchnorm_forward_test(X_in, gamma, beta, eps, mu, var)
    Z = np.tanh(X_out)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis = 1, keepdims = True)
    return Y, Z


def batchnorm_forward_test(x, gamma, beta, eps, mu, var):

    N, D = x.shape
    xmu = x - mu
    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar
    gammax = gamma * xhat
    out = gammax + beta
    
    return out, cache

def batchnorm_backward(dout, cache):

    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    N,D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. /(sqrtvar**2) * divar
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
    dsq = 1. /N * np.ones((N,D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. /N * np.ones((N,D)) * dmu
    dx = dx1 + dx2
    
    return dx, dgamma, dbeta

def gradW1(Xtrain, gB_norm):
    return Xtrain.T.dot(gB_norm)    

def gradb1(gB_norm):
    return (gB_norm).sum(axis = 0)
            
def gradB_norm(y_train_ind, Out_train, hidden_train, W2, cache):
    dout = (y_train_ind - Out_train).dot(W2.T)*(1-hidden_train**2)
    return batchnorm_backward(dout, cache)

def gradW2(y_train_ind, Out_train, hidden_train):
    return hidden_train.T.dot(y_train_ind - Out_train)  

def gradb2(y_train_ind, Out_train):
    return (y_train_ind - Out_train).sum(axis = 0)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))
eps = 10e-10
gamma = 1
beta = 0

N, D = x_train.shape
y_train_ind = ut.y2indicator(y_train)
y_test_ind = ut.y2indicator(y_test)
K = y_train_ind.shape[1]
M = 25

W1 = np.random.randn(D,M) / np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)
W1_copy = W1.copy()
W2_copy = W2.copy()

learning_rate = 10e-6
reg = 0.01
n_iters = 20

test_class_errors = []
Costs_test_nom = []
batch_size = 500
num_batches = int(np.ceil(N / batch_size))
decay = 0.99
mu = 0
var = 0

# batch GD no momentum

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = x_train[j * batch_size:(j + 1) * batch_size,:]
        Yb = y_train_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train, X_out, cache, b_mu, b_var = feedforward(Xb, W1, b1, W2, b2, gamma, beta, eps)
        mu = decay * mu + (1-decay)*b_mu
        var = decay * var + (1-decay)*b_var
    
        gW2 = gradW2(Yb, Out_train, hidden_train)
        gb2 = gradb2(Yb, Out_train)
        gB_norm, gGamma, gBeta = gradB_norm(Yb, Out_train, hidden_train, W2, cache)
        gW1 = gradW1(Xb, gB_norm)
        gb1 = gradb1(gB_norm)
        
        W2 += learning_rate*(gW2 - reg*W2)
        b2 += learning_rate*(gb2 - reg*b2)
        W1 += learning_rate*(gW1 - reg*W1)
        b1 += learning_rate*(gb1 - reg*b1)
        gamma += learning_rate*(gGamma)
        beta += learning_rate*(gBeta) 
        
        Out_test, hidden_test = feedforward_test(x_test, W1, b1, W2, b2, gamma, beta, eps, mu, var)          
        Cost_test = ut.cost(y_test_ind, Out_test)
        Costs_test_nom.append(Cost_test) 

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

print("Final error rate:", ut.error_rate(Out_test, y_test))
print("Final cost:", Costs_test_nom[-1])

plt.plot(Costs_test_nom)
plt.legend(['Test'])
plt.show()
