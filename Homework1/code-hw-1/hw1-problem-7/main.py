#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:54:14 2017
Code for Problem 7
@author: arcturus
"""
#%%
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy as sp
import gc

#%%
def load_dataset(): # this function taken from the assignment verbatim
    mndata = MNIST('python-mnist-0.3/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

#%%
def vectorize_labels(labels): # i wonder if there's a better way to do this
    num_classes = 10 #this is given to us, so we can hard-code it. 
    num_data_pts = labels.size
    matrix_labels = np.zeros((num_data_pts, num_classes))
    for i in range(0, num_data_pts):
        label_value = labels[i]
        matrix_labels[i,label_value] = 1
    return matrix_labels

#%%
def train(X, y, lam):
    num_classes = y.shape[1]
    data_dim = X.shape[1]
    W_hat = np.empty((data_dim, num_classes))
    xxT = X.transpose().dot(X)
    xyT = X.transpose().dot(y)
    I = np.identity(data_dim)
    W_hat = sp.linalg.solve((xxT + lam*I), xyT) #because never invert
    return W_hat

#%%
def predict(W, X):
    #W: R^(d X k), X: R^(m X d)
    return np.argmax(W.transpose().dot(X.transpose()), axis = 0) #very cute function

#%% 
def cross_validate(data, labels, data_test, labels_test):
    #%% Partition into random 80/20 train/test sets
    indices = np.random.permutation(data.shape[0])    
    tr_size = np.int(np.round(data.shape[0]*.8))
    cv_tr_idx, cv_test_idx = indices[:tr_size], indices[tr_size :]
    cv_tr_data, cv_test_data = data[cv_tr_idx, :], data[cv_test_idx, :]
    cv_tr_labels, cv_test_labels = labels[cv_tr_idx], labels[cv_test_idx]
    
    #%% 
    orig_data_dim = data.shape[1]
    #p = np.array([800, 900, 1000, 1150, 1270, 1380, 1490, 1650, 1800, 2000])# 3200, 4300, 5000, 5400])#, 1350, 1400, 1450])
    p = np.array([2000])
    err_cv_train = np.empty(p.size)
    err_cv_test = np.empty(p.size)
    #%% 
    for i in range(0, p.size):
        print("Size of data set is ", p[i])
        #%%
        G = np.sqrt(.1)*np.random.randn(p[i], orig_data_dim)
        b = np.random.uniform(0, 2*np.pi, p[i])
        B_tr_int = np.matlib.repmat(b, cv_tr_data.shape[0], 1)
        B_test_int = np.matlib.repmat(b, cv_test_data.shape[0], 1)
        B_real_test = np.matlib.repmat(b, data_test.shape[0], 1)
        
        #%% data transformed into higher dim
        H_train = np.cos(G.dot(cv_tr_data.transpose()) + B_tr_int.transpose())
        H_test = np.cos(G.dot(cv_test_data.transpose()) + B_test_int.transpose())
        H_real_test = np.cos(G.dot(data_test.transpose()) + B_real_test.transpose())
                             
        #%%
        cv_tr_mat_labels = vectorize_labels(cv_tr_labels)
        W_hat_p = train(H_train.transpose(), cv_tr_mat_labels, .0001)
        
        #%%
        pred_labels_train = predict(W_hat_p, H_train.transpose())
        pred_labels_test = predict(W_hat_p, H_test.transpose())
        pred_labels_real_test = predict(W_hat_p, H_real_test.transpose())
        #%%
        err_cv_train[i] = (1/cv_tr_labels.shape[0])*np.count_nonzero(pred_labels_train - cv_tr_labels)
        err_cv_test[i] = (1/cv_test_labels.shape[0])*np.count_nonzero(pred_labels_test - cv_test_labels)
        err_real_test = (1/data_test.shape[0])*np.count_nonzero(pred_labels_real_test - labels_test)
        #%%
        gc.collect()
    #%%
    return err_cv_train, err_cv_test, p, W_hat_p, G, err_real_test

#%%
if __name__ == '__main__':
    [X_train, labels_train, X_test, labels_test] = load_dataset() #part 0
    y = vectorize_labels(labels_train) # part b
    W_hat = train(X_train, y, .0001) # part c

    #%% part c continued
    predicted_labels_train = predict(W_hat, X_train)
    predicted_labels_test = predict(W_hat, X_test)

    #%%
    accu_train = 1 - (1/labels_train.shape[0])*np.count_nonzero(predicted_labels_train - labels_train)
    accu_test = 1 - (1/labels_test.shape[0])*np.count_nonzero(predicted_labels_test  - labels_test)
    
    #%% part d, e, cross validation and running on test data
    [err_train, err_test, p, W_hat_p, G, err_real_test] = cross_validate(X_train, labels_train, X_test, labels_test)

    #%% 
    plt.plot(p, err_train)
    plt.plot(p, err_test)
    plt.legend(['training error', 'test error'])
    plt.xlabel('p values (# dimensions in transformed features)')
    plt.ylabel('error (as a fraction)')
    plt.title('Cross Validation: How Many Transformed Features Do We Need?')
    
