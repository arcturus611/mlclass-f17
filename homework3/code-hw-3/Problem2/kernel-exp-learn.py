#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:57:32 2017

@author: arcturus
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import seaborn as sns

#%%
def compute_fx(x):
    fx = 4*np.sin(np.pi*x)*np.cos(6*np.pi*np.square(x))
    return fx

#%%
def predict_x_exp(train, test, gamma, alpha_hat):
    # Input: train points, test point, gamma(for exp kernel), model alpha
    return np.inner(alpha_hat, generate_exp_kernel(test, train, gamma))

#%% 
def gen_data(n):
    x = np.random.rand(n) #use this for testing: x = np.linspace(0, 1, 300)
    fx = compute_fx(x)#4*np.sin(np.pi*x)*np.cos(6*np.pi*np.square(x))
    eps =  np.random.randn(n)#additive gaussian noise
    y = fx + eps
    return x, y, fx

#%% 
def generate_exp_kernel(X, Y, gamma):
    # given "vectors" X, Y and param gamma, we compute [K_{ij}]= (exp(-gamma*(x_i - x_j)^2)
    # Input: X (dims1), Y (dims2), gamma (natural number)
    # Outout: K (dims1 X dims2) matrix corr. to above kernel
    # WARNING: First input must be either vector (for training phase)
    # OR, a scalar (for testing)
    if(type(X)!=np.ndarray):
        K_exp = np.exp(-gamma*np.square(X - Y[None, :]))
        K_exp = K_exp[0]
        #print("Not array!")
    else:
        #print("array!")
        K_exp = np.exp(-gamma*np.square(X[:, None] - Y[None, :]))
    return K_exp

#%%
def gamma_choose_heuristic(X):
    # For the rbf kernel, we have a heuristic to choose gamma which helps to narrow down the range (is that what that means?)
    # Input: Data x
    # Output: huristic gamma
    
    #WARNING: input MUST be a vector of len > 1
    
    dists = np.square(X[:, None] - X[None, :])
    dists_vec = np.reshape(dists, len(X)*len(X), 1)
    heu_gamma = 1/np.median(dists_vec)
    return heu_gamma

#%% 
def solve_for_alpha(K, lam, y):
    # Stick in the solution for least squares 
    n = len(y) #number of coordintes in y
    alpha = np.linalg.solve(K + lam*np.identity(n), y)
    return alpha

#%% 
def compute_pred_error(pred_val, true_val):
    err = np.power(pred_val - true_val, 2)
    return err

#%% 
def leave_one_out_of_vec(X, idx):
    # return vector with element at X(idx) removed, and also that removed elemnt
    mask = np.ones(len(X), dtype = bool)
    mask[idx] = False
    vec = X[mask]
#    elem = X[~mask]
#    elem = elem[0] 
    elem = X[idx]
    # return scalar, coz that's what makes sense, even if this code ends up being a bit ugly
    # (ugliness can't be hidden, only transferred: law of conservation of ugliness.)
    return vec, elem
#%% 
def leave_one_out_of_mat(X, idx):
    # Input: matrix X, idx for which row and col are to be removed
    # Output: matrix Z, submatrix from X, removing idx-th row and col
    # Assuming it's a square matrix (of course)
    
    mask = np.ones(X.shape[0], dtype = bool)
    mask[idx] = False
    Z = X[np.ix_(mask, mask)]
    return Z

#%% 
def learn_exp_kernel_by_cv(lam, gamma, x, y, fx):
    # NOTE: for now, the two kernels are separate. But later, IF NEED BE, we can write one common thing for both. 
    # Input: lam and d are hyperparams; x is vector of input data and y are corr true fn vals
    # Output: total error
    # We use "unused" to denote unused element 
    #%%
    K = generate_exp_kernel(x, x, gamma)
    #print("K with d = {} and lam = {} is {}".format(d, lam, K)) #XKCD
    total_err = 0
    #%%
    for i in range(len(x)):
        #%% 
        # Remember,i is the one you are leaving out (test data) 
        K_tr = leave_one_out_of_mat(K, i)
        (y_tr, y_test) = leave_one_out_of_vec(y, i) 
        fx_test = fx[i] #true (noiseless val of fn on x_test)
        #rememeber, we ned to use y, not fx, for training (noise incl)
        # also, remmber, y_test is bakwas. we DON'T TOUCH IT 
        #%% 
        alpha_hat = solve_for_alpha(K_tr, lam, y_tr)
        (data_tr, data_test) = leave_one_out_of_vec(x, i)
        pred_test = predict_x_exp(data_tr, data_test, gamma, alpha_hat)
        total_err+= compute_pred_error(pred_test, fx_test)
    #%%
    return total_err
#%% 
if __name__ == '__main__':
    #%%
    n = 30 #XKCD
    # Okay, based on FORTY test runs, I conclude: best_lam= .1, best_d = 36. Cool!
    (x, y, fx) = gen_data(n)
    lam = np.linspace(.01, 10, 15) 
    heu_gamma = gamma_choose_heuristic(x)
    print("heuristic computed value is {}".format(heu_gamma))
    gamma = np.linspace(heu_gamma-1, heu_gamma+60, 100) #400
    total_err = np.empty((len(lam), len(gamma)))
    #%%learn_poly_kernel_by_cv(lam_grid, d_grid, x, fx)
    for i in np.arange(len(lam)):
        print("i = {}".format(i))
        for j in np.arange(len(gamma)):
            total_err[i, j] = learn_exp_kernel_by_cv(lam[i], gamma[j], x, y, fx)
            
    #%% based on total_err, we find the best lam, d
    (lam_min_idx, gamma_min_idx)= np.unravel_index(total_err.argmin(), total_err.shape)
    best_lam = lam[lam_min_idx]
    best_gamma = gamma[gamma_min_idx]
    print("(best_lam = {},  best_gamma = {}), and equals total_err[best_lam, best_gamma]= {}".format(best_lam, best_gamma, total_err[lam_min_idx, gamma_min_idx]))
    
    #%% Now we build the predictor
    K_learnt = generate_exp_kernel(x, x, best_gamma)
    alpha_learnt = solve_for_alpha(K_learnt, best_lam, y)
    
    #%% final stuff (for part b)
    sns.set()
    
    total_err = learn_exp_kernel_by_cv(best_lam, best_gamma, x, y, fx)
    plt.figure(1)
    plt.scatter(x, y, c='k')
    
    num_new_test = 200
    new_test = np.linspace(0, 1, num_new_test)
    fx_new_test = compute_fx(new_test)
    fxhat_new_test = np.empty(num_new_test)
    for i in range(num_new_test):
        fxhat_new_test[i] = np.inner(alpha_learnt, generate_exp_kernel(new_test[i], x, best_gamma))

    plt.plot(new_test, fx_new_test, c='b')
    plt.plot(new_test, fxhat_new_test, c = 'r')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([-4.5, 4.5])
    plt.legend(['true fit', 'learnt fit', 'data'])
    plt.title('Learning kernel hyperparameters (exponential)')