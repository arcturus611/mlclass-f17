#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:09:15 2017

@author: arcturus
"""
#%% get started with yelp-ing
import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
#%% How to import fn from file in diff dir.? Need to know. 

#%% coordinate descent for lasso
def fast_coord_des_lasso(X, y, w_init, lam):
    #%% init to enter while loop
    (delta, tol, count) = (10, .3, 0)
    (n, d) = X.shape
    w = w_init
    
    #%% pre-calc stuff
    a = 2*np.square(la.norm( X, ord=2, axis=0 ))
    Xw = np.dot(X, w)
    Xw_less = Xw - w[d-1]*X[:, d-1, np.newaxis] #first time, remove last weighted col
    
    #%% while loop
    while (delta>tol):
        #%%
        (count, delta) = (count + 1, 0)
        Xw = Xw_less + w[d-1]*X[:, d-1, np.newaxis]
        w0 = np.mean( y - Xw )
        Xy_adj = np.dot(X.T, y - w0) #<-- XKCD

        #%%
        for curr in range(0, d):
            #%%
            if (curr==0):
                prev = d-1
            else:
                prev = curr-1
                
            #%%
            Xw_col_prev = w[prev]*X[:, prev, np.newaxis]
            X_col_curr = X[:, curr, np.newaxis]
            Xw_col_curr = w[curr]*X_col_curr
            Xw_less = Xw_less + Xw_col_prev - Xw_col_curr
            
            #%%           
            c_k = 2*(Xy_adj[curr] - np.dot(X_col_curr.T, Xw_less))
            
            #%%
            if (c_k < -lam): 
                val = (c_k + lam)/a[curr]
            elif (c_k > lam):
                val = (c_k - lam)/a[curr]
            else:
                val = 0
            #%%
            delta = np.max((delta, np.abs(w[curr] - val)))
            #%%
            w[curr] = val
    #%% 
    return w

#%% check objective (this can be called INSIDE lasso to ensure obj is decr.; not necessary, jiust as a sanity chk)
def eval_lasso_loss(X, w, w0, y, lam):
    loss = np.square(la.norm(np.dot(X, w) + w0 - y, 2)) + lam*la.norm(w, 1)
    return loss

#%% lambda_max
def gen_lam_max(X, y):
    (n, d) = X.shape
    Y = np.matlib.repmat(y, 1, d)
    XY = X*Y #yes, i intend to do pointwise mult
    sum_XY_cols = np.sum(XY, axis = 0)
    sum_y = np.sum(y)/n
    sum_X_cols = np.sum(X, axis = 0)
    lam = 2*np.abs(np.max(sum_XY_cols - sum_y*sum_X_cols))
    return lam

#%% validation error
def compute_val_err(y, X, w):
    return np.mean((y - np.dot(X, w))**2) 

#%% main function!
if __name__ == '__main__':
    #%% load data, max lambda, then repeat: run lasso + evaluate it
    # X: n obs of d dims each; w0: offset of n dims; y: measured signal
    X = np.genfromtxt("yelp_data/upvote_data.csv", delimiter=",")
    y = np.loadtxt("yelp_data/upvote_labels.txt", dtype=np.int)
    featureNames = open("yelp_data/upvote_features.txt").read().splitlines()
   
    #%% generate train, validation and test sets
    X_train = X[0: 4000, :]
    y_train = y[0: 4000, np.newaxis]
    X_valid = X[4000: 5000, :]
    y_valid = y[4000: 5000, np.newaxis]
    X_test = X[5000: , :]
    y_test = y[5000: , np.newaxis]
    
    #%% 
    num_lasso_runs = 10 #XKCD for now. later change to while
    lam = np.zeros(num_lasso_runs+1)
    valid_err = np.zeros(num_lasso_runs)
    loss_fast = np.zeros(num_lasso_runs)
    (n, d) = X.shape 
    w_init_fast = np.zeros((d, 1))#np.zeros((d, 1)) #HORRIFYING OSERVATION: random.randn doesn't work at all. This is terrible. 
    
    #%%
    lam[0] = gen_lam_max(X_train, y_train) 
    for i in range(0, num_lasso_runs): 
        w_hat_fast = fast_coord_des_lasso(X_train, y_train, w_init_fast, lam[i])
        loss_fast[i] = compute_val_err(y_train, X_train, w_hat_fast)
        valid_err[i] = compute_val_err(y_valid, X_valid, w_hat_fast)
        print('~~The lossfast and v_error with lamda= ' + str(lam[i]) + ' are ' + str(loss_fast) + ' and ' + str(valid_err[i]) + ' \n')
        lam[i+1] = lam[i]/2
        w_init_fast = w_hat_fast
        
        #%%
    plt.figure(1)
    plt.plot(lam[0:num_lasso_runs], loss_fast)
    plt.plot(lam[0:num_lasso_runs], valid_err)
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.title('Validation/Training Error vs lambda')
    plt.legend(['Training Err', 'Validation Err'])
        
    y_pred_fast = np.dot(X_test, w_hat_fast)
        #% Write more code to predict the stuff (one line)