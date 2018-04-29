#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:48:23 2017

@author: arcturus
"""

#%% import libraries
import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt

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

#%% validation error
def compute_val_err(y, X, w):
    return np.mean((y - np.dot(X, w))**2) 

#%% check objective
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

#%% generate synthetic data
def gen_syn_data(n, d, k, sigma):
    X = np.random.randn(n, d)
    eps = sigma*np.random.randn(n, 1)
    wstar = 10*(2*np.random.randint(low=0, high=2, size=(d, 1)) - np.ones((d, 1)))
    wstar[k:] = 0 #zero-ing out stuff from kth index (remember, we are starting at 0th index)
    w0 = 0*np.ones((n, 1))
    y = np.dot(X, wstar) + eps + w0
    return X, wstar, w0, y

#%% main function!
if __name__ == '__main__':
   #%% given input parameters 
    (n, d, k, sigma, num_lasso_runs) = (50, 75, 5, 1, 10)
    
    #%% generate synthetic data, max lambda, then repeat: run lasso + evaluate it
    # X: n obs of d dims each; wstar: d dims true model vector; w0: offset of n dims
    [X, wstar, w0, y] = gen_syn_data(n, d, k, sigma)
    lam = np.zeros(num_lasso_runs+1)
    precision = np.zeros(num_lasso_runs)
    recall = np.zeros(num_lasso_runs)
    lam[0] = gen_lam_max(X, y)
    w_init = np.ones((d, 1)) #init to zeros --> errors due to how lasso is written
    w_init_fast = np.random.randn(d, 1)
    for i in range(0, 10): 
         w_hat_fast = fast_coord_des_lasso(X, y, w_init_fast, lam[i])             
         loss = compute_val_err(y, X, w_hat_fast)
         print('~~The loss with lamda= ' + str(lam[i]) + ' is ' + str(loss))
         precision[i] =  sum(np.logical_and(w_hat_fast, wstar))/np.count_nonzero(w_hat_fast)
         recall[i] = sum(np.logical_and(w_hat_fast, wstar))/k
         lam[i+1] = lam[i]/2
         w_init_fast = w_hat_fast
         #w_init = w_hat
         
    #%% plot stuff
    plt.figure(1)
    plt.plot(lam[0: num_lasso_runs], precision, 'ro')
    plt.xlabel('lambda')
    plt.ylabel('precision')
    plt.title('precision vs lambda')
    
    plt.figure(2)
    plt.plot(lam[0: num_lasso_runs], recall, 'bo')
    plt.xlabel('lambda')
    plt.ylabel('recall')
    plt.title('recall vs lambda')
    
#    