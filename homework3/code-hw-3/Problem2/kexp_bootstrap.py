#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:57:32 2017

@author: arcturus
"""

#%%
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from scipy import stats
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
def solve_for_alpha(K, lam, y):
    # Stick in the solution for least squares 
    n = len(y) #number of coordintes in y
    alpha = np.linalg.solve(K + lam*np.identity(n), y)
    return alpha


#%% 
if __name__ == '__main__':
    #%%
    n = 30 
    # Okay, based on FORTY test runs, I conclude: best_lam= .1, best_gamma = 36. Cool!
    (x, y, fx) = gen_data(n)
    
    num_new_test = 200
    num_bootstrap = 300
    new_test = np.linspace(0, 1, num_new_test)
    fx_new_test = compute_fx(new_test)
    fxhat_new_test = np.empty((num_bootstrap, num_new_test))
    #%%     
#    lam = np.linspace(0, 1.5, 15) 
#    d = np.linspace(30, 42, 200) #400
    best_lam = .01
    best_gamma = 40
    #%%
    for kk in np.arange(num_bootstrap):
        sample_indices = np.random.choice(30, 30, replace = True)
        x_sampled = x[sample_indices]
        y_sampled = y[sample_indices]
        fx_sampled = fx[sample_indices]

        #%% Now we build the predictor
        K_learnt = generate_exp_kernel(x_sampled, x_sampled, best_gamma)
        alpha_learnt = solve_for_alpha(K_learnt, best_lam, y_sampled)
        

        for i in range(num_new_test):
            fxhat_new_test[kk, i] = np.inner(alpha_learnt, generate_exp_kernel(new_test[i], x_sampled, best_gamma))
    
    #%%
    eighth_largest_elem = np.empty(num_new_test)
    eighth_smallest_elem = np.empty(num_new_test)
    
    for i in np.arange(num_new_test):
        col = fxhat_new_test[:, i]
        eighth_largest_elem[i] = np.partition(col.flatten(), -8)[-8]
        eighth_smallest_elem[i] = np.partition(col.flatten(), 7)[7]
    
    #%%
    orig_best_kernel = generate_exp_kernel(x, x, best_gamma)
    orig_best_alpha = solve_for_alpha(orig_best_kernel, best_lam, y)
    fxhat_orig_new_test = np.empty(num_new_test)
    for i in range(num_new_test):
        fxhat_orig_new_test[i] = np.inner(orig_best_alpha, generate_exp_kernel(new_test[i], x, best_gamma))
    
    #%%
    fig, ax = plt.subplots(1)
    plt.scatter(x, y, c='k')
    plt.plot(new_test, fx_new_test, c='b')
    plt.plot(new_test, fxhat_orig_new_test, c = 'r')
    ax.fill_between(new_test, eighth_smallest_elem, eighth_largest_elem, facecolor='blue', alpha = .1)
    ax.set_xlim([0,1])
    ax.set_ylim([-4.5, 4.5])
    
    plt.legend(['true fit', 'learnt fit', 'data', '95% confidence interval'])
    plt.title('Bootstrapping (exp kernel)')