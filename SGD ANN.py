# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:42:36 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import utils as ut
from datetime import datetime

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
last_dt = 0
intervals = []

learning_rate = 0.9
reg = 0.01
n_iters = 101

Costs_test = []
test_class_errors = []
t0 = datetime.now()
last_dt = 0
# full
for epoch in range(n_iters):
    Out_train, hidden_train = ut.feedforward(Xtrain, W1, b1, W2, b2)
    Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
    
    gW2 = ut.gradW2(Ytrain_ind, Out_train, hidden_train) / N
    gb2 = ut.gradb2(Ytrain_ind, Out_train) / N
    gW1 = ut.gradW1(Xtrain, Ytrain_ind, Out_train, hidden_train, W2) / N
    gb1 = ut.gradb1(Ytrain_ind, Out_train, hidden_train, W2) / N
    
    W2 += learning_rate*(gW2 - reg*W2)
    b2 += learning_rate*(gb2 - reg*gb2)
    W1 += learning_rate*(gW1 - reg*W1)
    b1 += learning_rate*(gb1 - reg*gb1)
    
    dt = (datetime.now() - t0).total_seconds()
    
    Cost_test = ut.cost(Ytest_ind, Out_test)
    Costs_test.append([dt, Cost_test])
    
    Rate_test = ut.error_rate(Out_test, Ytest)
    test_class_errors.append([dt, Rate_test])
    
    dt2 = dt - last_dt
    last_dt = dt
    intervals.append(dt2)
    
    if (epoch + 1) % 10 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))
        
Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)        
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test[-1][1])
print("Elapsed time for full GD", datetime.now() - t0)
        
max_dt = dt
avg_interval_dt = np.mean(intervals)
        
# stohastic

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)

Costs_test_sgd = []
learning_rate = 0.001
reg = 0.01  

t0 = datetime.now()
last_calculated_loss = 0
done = False

for epoch in range(n_iters):
    tmpX = Xtrain
    tmpY_ind = Ytrain_ind

    for n in range(N):
        x = tmpX[n,:].reshape(1,D)
        y = tmpY_ind[n,:].reshape(1,K)

        Out_train, hidden_train = ut.feedforward(x, W1, b1, W2, b2)
        gW2 = ut.gradW2(y, Out_train, hidden_train)
        gb2 = ut.gradb2(y, Out_train)
        gW1 = ut.gradW1(x, y, Out_train, hidden_train, W2)
        gb1 = ut.gradb1(y, Out_train, hidden_train, W2)
            
        W2 += learning_rate*(gW2 - reg*W2)
        b2 += learning_rate*(gb2 - reg*gb2)
        W1 += learning_rate*(gW1 - reg*W1)
        b1 += learning_rate*(gb1 - reg*gb1)
        
        dt = (datetime.now() - t0).total_seconds()
        dt2 = dt - last_calculated_loss
        
        if dt2 > avg_interval_dt:
            last_calculated_loss = dt
            Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
            Cost_test = ut.cost(Ytest_ind, Out_test)
            Costs_test_sgd.append([dt, Cost_test])
        
        if dt > max_dt:
            done = True
            break
    if done:
        break
    
    if (epoch + 1) % 1 == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_sgd[-1][1])
print("Elapsed time for SGD", datetime.now() - t0)

plt.plot(np.array(Costs_test_sgd)[:,1])
plt.plot(np.array(Costs_test)[:,1])

# batch GD

W1 = W1_copy.copy()
W2 = W2_copy.copy()
b1 = np.zeros(M)
b2 = np.zeros(K)

Costs_test_bgd = []
learning_rate = 0.9
reg = 0.01
batch_size = 500

t0 = datetime.now()
last_calculated_loss = 0
done = False

for epoch in range(n_iters):
    tmpX = Xtrain
    tmpY_ind = Ytrain_ind
    num_batches = int(np.ceil(N / batch_size))
    
    for j in range(num_batches):
        Xb = tmpX[j * batch_size:(j + 1) * batch_size,:]
        Yb = tmpY_ind[j * batch_size:(j + 1) * batch_size,:]
        current_batch_sz = len(Xb)

        Out_train, hidden_train = ut.feedforward(Xb, W1, b1, W2, b2)
        
        gW2 = ut.gradW2(Yb, Out_train, hidden_train) / current_batch_sz
        gb2 = ut.gradb2(Yb, Out_train) / current_batch_sz
        gW1 = ut.gradW1(Xb, Yb, Out_train, hidden_train, W2) / current_batch_sz
        gb1 = ut.gradb1(Yb, Out_train, hidden_train, W2) / current_batch_sz
        
        W2 += learning_rate*(gW2 - reg*W2)
        b2 += learning_rate*(gb2 - reg*gb2)
        W1 += learning_rate*(gW1 - reg*W1)
        b1 += learning_rate*(gb1 - reg*gb1)
        
        dt = (datetime.now() - t0).total_seconds()
        dt2 = dt - last_calculated_loss
        
        if dt2 > avg_interval_dt:
            last_calculated_loss = dt
            Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)          
            Cost_test = ut.cost(Ytest_ind, Out_test)
            Costs_test_bgd.append([dt, Cost_test])
        
        if dt > max_dt:
            done = True
            break
    if done:
        break
    
    if (epoch + 1) % num_batches == 0:
        print("Cost at iteration %d: %.6f" % (epoch + 1, Cost_test))

Out_test, hidden_test = ut.feedforward(Xtest, W1, b1, W2, b2)
print("Final error rate:", ut.error_rate(Out_test, Ytest))
print("Final cost:", Costs_test_sgd[-1][1])
print("Elapsed time for BGD", datetime.now() - t0)

plt.plot(np.array(Costs_test_sgd)[:,0], np.array(Costs_test_sgd)[:,1], label="SGD")
plt.plot(np.array(Costs_test)[:,0], np.array(Costs_test)[:,1], label="FGD")
plt.plot(np.array(Costs_test_bgd)[:,0], np.array(Costs_test_bgd)[:,1], label="BGD")
plt.legend()
plt.show()
