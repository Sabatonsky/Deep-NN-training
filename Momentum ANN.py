# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:42:36 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import utils as ut

Xtrain, Xtest, Ytrain, Ytest = ut.get_normalized_data()

N, D = Xtrain.shape
Ytrain_ind = ut.y2indicator(Ytrain)
Ytest_ind = ut.y2indicator(Ytest)
K = Ytrain_ind.shape[1]
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
Costs_test_comm = []
Costs_test_nesm = []
batch_size = 500
num_batches = int(np.ceil(N / batch_size))
mu = 0.90

# batch GD no momentum

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = Xtrain[j * batch_size:(j + 1) * batch_size,:]
        Yb = Ytrain_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train)
        gb2 = ut.gradb2(Yb, Out_train)
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        W2 += learning_rate*(gW2 - reg*W2)
        b2 += learning_rate*(gb2 - reg*b2)
        W1 += learning_rate*(gW1 - reg*W1)
        b1 += learning_rate*(gb1 - reg*b1)
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_nom.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_nom[-1])

# batch GD momentum

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)
gW2 = 0
gb2 = 0
gW1 = 0
gb1 = 0
vW2 = 0
vb2 = 0
vW1 = 0
vb1 = 0
tmpX = Xtrain
tmpY_ind = Ytrain_ind

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        vW2 = mu*vW2 - learning_rate*ut.gradW2(Yb, Out_train, hidden_train)
        vb2 = mu*vb2 - learning_rate*ut.gradb2(Yb, Out_train)
        vW1 = mu*vW1 - learning_rate*ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        vb1 = mu*vb1 - learning_rate*ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        W2 -= vW2 + learning_rate*reg*W2
        b2 -= vb2 + learning_rate*reg*b2
        W1 -= vW1 + learning_rate*reg*W1
        b1 -= vb1 + learning_rate*reg*b1
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_comm.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_comm[-1])

# batch GD система Нестерова

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)
gW2 = 0
gb2 = 0
gW1 = 0
gb1 = 0
vW2 = 0
vb2 = 0
vW1 = 0
vb1 = 0
tmpX = Xtrain
tmpY_ind = Ytrain_ind

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train)
        gb2 = ut.gradb2(Yb, Out_train)
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        vW2 = mu*vW2 - learning_rate*gW2
        vb2 = mu*vb2 - learning_rate*gb2
        vW1 = mu*vW1 - learning_rate*gW1
        vb1 = mu*vb1 - learning_rate*gb1
        
        W2 -= mu*vW2 - learning_rate*(gW2 - reg*W2)
        b2 -= mu*vb2 - learning_rate*(gb2 - reg*b2)
        W1 -= mu*vW1 - learning_rate*(gW1 - reg*W1)
        b1 -= mu*vb1 - learning_rate*(gb1 - reg*b1)
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_nesm.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_nesm[-1])

plt.plot(Costs_test_nom, label="no momentum")
plt.plot(Costs_test_comm, label="momentum")
plt.plot(Costs_test_nesm, label="Nesterov momentum")
plt.legend()
plt.show()
