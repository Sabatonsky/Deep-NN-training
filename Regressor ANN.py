# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:47:52 2023

@author: Bannikov Maxim
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

def classification_rate(Y,P):
    return np.mean(Y == P)

def cost(Y, T):
    return np.mean((T - Y)**2)

N = 500
X = np.random.random((N, 2))*4 - 2
Y = X[:,0]*X[:,1]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

D = 2
M = 100

W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

V = np.random.randn(M) / np.sqrt(M)
c = 0

def forward(X):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)
    Yhat = Z.dot(V) + c
    return Z, Yhat

lr = 10e-6
costs = []

for epoch in range(20001):
    Z, Yhat = forward(X)
    cY = cost(Yhat, Y)
    costs.append(cY)
    if epoch % 1000 == 0:
        print("Epoch number:", epoch, "Cost:", c)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X[:,0],X[:,1],Yhat)
        plt.show()
    V += lr * Z.T.dot(Y - Yhat)
    c += lr * (Y - Yhat).sum()
    W += lr * X.T.dot(np.outer(Y - Yhat, V) * (Z > 0))
    b += lr * (np.outer(Y - Yhat, V) * (Z > 0)).sum()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0],X[:,1], Y) 
   
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)    
plt.show()

Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)
plt.scatter(Xgrid[:,0], Xgrid[:,1], c = R)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_trisurf(Xgrid[:,0],Xgrid[:,1], R, linewidth = 0.2, antialiased = True)
plt.show()
