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

learning_rate = 0.00004
reg = 0.01
n_iters = 10

test_class_errors = []
Costs_test_cls = []
Costs_test_RMSP = []
Costs_test_AG = []
Costs_test_Adam = []
batch_size = 500
num_batches = int(np.ceil(N / batch_size))
mu = 0.90

# batch GD fixed learning rate

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
        Costs_test_cls.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_cls[-1])

# batch GD AdaGrad

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)
gW2 = 0
gb2 = 0
gW1 = 0
gb1 = 0
tmpX = Xtrain
tmpY_ind = Ytrain_ind
cacheW2 = np.zeros((M,K))
cacheb2 = np.zeros(K)
cacheW1 = np.zeros((D,M))
cacheb1 = np.zeros(M)
lr0 = 0.001

eps = 10e-8

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train)
        gb2 = ut.gradb2(Yb, Out_train)
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        cacheW2 += gW2**2
        cacheb2 += gb2**2
        cacheW1 += gW1**2
        cacheb1 += gb1**2
    
        W2 += lr0*(gW2 - reg*W2) / np.sqrt(cacheW2 + eps)
        b2 += lr0*(gb2 - reg*b2) / np.sqrt(cacheb2 + eps)
        W1 += lr0*(gW1 - reg*W1) / np.sqrt(cacheW1 + eps)
        b1 += lr0*(gb1 - reg*b1) / np.sqrt(cacheb1 + eps)
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_AG.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_AG[-1])

# batch GD RMSProp

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)
gW2 = 0
gb2 = 0
gW1 = 0
gb1 = 0
tmpX = Xtrain
tmpY_ind = Ytrain_ind
beta = 0.99

cacheW2 = np.ones((M,K))
cacheb2 = np.ones(K)
cacheW1 = np.ones((D,M))
cacheb1 = np.ones(M)

for epoch in range(n_iters): 
    for j in range(num_batches):
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train)
        gb2 = ut.gradb2(Yb, Out_train)
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        cacheW2 = beta * cacheW2 + (1 - beta) * gW2**2
        cacheb2 = beta * cacheb2 + (1 - beta) * gb2**2
        cacheW1 = beta * cacheW1 + (1 - beta) * gW1**2
        cacheb1 = beta * cacheb1 + (1 - beta) * gb1**2
    
        W2 += lr0*(gW2 - reg*W2) / np.sqrt(cacheW2 + eps)
        b2 += lr0*(gb2 - reg*b2) / np.sqrt(cacheb2 + eps)
        W1 += lr0*(gW1 - reg*W1) / np.sqrt(cacheW1 + eps)
        b1 += lr0*(gb1 - reg*b1) / np.sqrt(cacheb1 + eps)
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_RMSP.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_RMSP[-1])

# batch GD Adam

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)
gW2 = 0
gb2 = 0
gW1 = 0
gb1 = 0
tmpX = Xtrain
tmpY_ind = Ytrain_ind
beta_1 = 0.9
beta_2 = 0.999

v_W2 = 0
v_b2 = 0
v_W1 = 0
v_b1 = 0

m_W2 = 0
m_b2 = 0
m_W1 = 0
m_b1 = 0
t = 0

for epoch in range(n_iters): 
    for j in range(num_batches):
        t += 1
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train)
        gb2 = ut.gradb2(Yb, Out_train)
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2)
        
        v_W2 = beta_2 * v_W2 + (1 - beta_2) * gW2**2
        v_b2 = beta_2 * v_b2 + (1 - beta_2) * gb2**2
        v_W1 = beta_2 * v_W1 + (1 - beta_2) * gW1**2
        v_b1 = beta_2 * v_b1 + (1 - beta_2) * gb1**2
        
        vhat_W2 = v_W2 / (1 - beta_2**t)
        vhat_b2 = v_b2 / (1 - beta_2**t)
        vhat_W1 = v_W1 / (1 - beta_2**t)
        vhat_b1 = v_b1 / (1 - beta_2**t)
        
        m_W2 = beta_1 * m_W2 + (1 - beta_1) * gW2
        m_b2 = beta_1 * m_b2 + (1 - beta_1) * gb2
        m_W1 = beta_1 * m_W1 + (1 - beta_1) * gW1
        m_b1 = beta_1 * m_b1 + (1 - beta_1) * gb1
        
        mhat_W2 = m_W2 / (1 - beta_1**t)
        mhat_b2 = m_b2 / (1 - beta_1**t)
        mhat_W1 = m_W1 / (1 - beta_1**t)
        mhat_b1 = m_b1 / (1 - beta_1**t)
        
        W2 += lr0*(mhat_W2 - reg*W2) / np.sqrt(vhat_W2 + eps)
        b2 += lr0*(mhat_b2 - reg*b2) / np.sqrt(vhat_b2 + eps)
        W1 += lr0*(mhat_W1 - reg*W1) / np.sqrt(vhat_W1 + eps)
        b1 += lr0*(mhat_b1 - reg*b1) / np.sqrt(vhat_b1 + eps)
        
        Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
        Cost_test = ut.cost(Ytest_ind, Out_test)
        Costs_test_Adam.append(Cost_test)

    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_Adam[-1])

plt.plot(Costs_test_cls, label='Common')
plt.plot(Costs_test_AG, label='AdaGrad')
plt.plot(Costs_test_RMSP, label='RMSProp')
plt.plot(Costs_test_Adam, label='Adam')
plt.legend()
plt.xlim(1000,1200)
plt.ylim(0,0.05)
plt.show()
