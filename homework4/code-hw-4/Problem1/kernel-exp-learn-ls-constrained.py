#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:51:54 2017

@author: arcturus
"""

# TODO
# 4) learn_kernel_.... function should be able to take var-lenght params? for var num of hyperparams?
#%%
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
def solve_for_alpha(K, lam1, y):
    n = len(y) 
    alpha = Variable(n)
    D = -1*np.eye(n) + np.eye(n, k = 1)
    D[n-1, n-1] = 1
    #this fixed it! 
    # remember, we want non-decreasing values of the learnt function, 
    # but we also don't want them all to be negative 
    # (as was being specified by this last guy being -1)
    K_mod = D.dot(K)    
    objective = Minimize(sum_squares(K*alpha - y) + lam1*quad_form(alpha, K))
    constraints = [0 <= K_mod*alpha] #just to check if it works at all (should return ls value) 
    prob = Problem(objective, constraints)
    result = prob.solve() 
    return alpha.value

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
    else:
        K_exp = np.exp(-gamma*np.square(X[:, None] - Y[None, :]))
    return K_exp 
#%% 
def leave_one_out_of_vec(X, idx):
    # return vector with element at X(idx) removed, and also that removed elemnt
    mask = np.ones(len(X), dtype = bool)
    mask[idx] = False
    vec = X[mask]
    elem = X[idx]
    # return scalar
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
def predict_x_exp(train, test, gamma, alpha_hat):
    # Input: train points, test point, gamma(for exp kernel), model alpha
    return np.inner(alpha_hat.T, generate_exp_kernel(test, train, gamma))

#%% generate data
def generate_data(n):
    x = (np.arange(n))/(n-1)
    fx = compute_fx(x)
    eps =  np.random.randn(n)#additive gaussian noise
    y = fx + eps
    y[15] = 0 #outlier 
    return x, fx, y
#%%compute f(x)
def compute_fx(x):
    return 10*(1*np.greater_equal(x, 1/5) + 1*np.greater_equal(x, 2/5) + 1*np.greater_equal(x, 3/5) + 1*np.greater_equal(x, 4/5))

#%% 
def compute_pred_error(pred_val, true_val):
    err = np.power(pred_val - true_val, 2)
    return err

#%% 
def learn_exp_kernel_by_cv(lambda_1, gamma, x, y, fx):
    # Input: lambda_1, lambda_2 and gamma are hyperparams; x is vector of input data and y are corrupt values and true fn vals
    # Output: total error over all cross-validation runs using these hyperparams
    # We use "unused" to denote unused element 
    #%%
    K = generate_exp_kernel(x, x, gamma)
    total_err = 0
    #%%
    for i in range(len(x)):
        #%% 
        # Remember,i is the one you are leaving out (test data) 
        K_tr = leave_one_out_of_mat(K, i)
        (y_tr, y_test) = leave_one_out_of_vec(y, i) 
        fx_test = fx[i] #true (noiseless val of fn on x_test)
        #rememeber, we ned to use y, not fx, for training (noise incl)
        # also, remmber, y_test is not what we use 
        #%% 
        alpha_hat = solve_for_alpha(K_tr, lambda_1, y_tr)
        (data_tr, data_test) = leave_one_out_of_vec(x, i)
        pred_test = predict_x_exp(data_tr, data_test, gamma, alpha_hat)
        total_err+= compute_pred_error(pred_test, fx_test)
    #%%
    return total_err
#%% 
if __name__ == '__main__':
    #%% input params, init stuff
    n = 30 
    (x, fx, y) = generate_data(n)
    lambda_1 = np.linspace(.01, 10, 10) 
    heu_gamma = gamma_choose_heuristic(x)
    print("heuristic computed value is {}".format(heu_gamma))
    gamma = np.linspace(heu_gamma-1, heu_gamma+30, 15) #400
    total_err = np.empty((len(lambda_1), len(gamma)))

    #%%learn the kernel for each hyperparam pair using loocv on input data
    for i in np.arange(len(lambda_1)):
        print("i = {}".format(i))
        for j in np.arange(len(gamma)):
            print("~~~ j = {}".format(j))
            total_err[i, j] = learn_exp_kernel_by_cv(lambda_1[i], gamma[j], x, y, fx)
                
    #%% based on total_err, we find the best lambda_1, d
    (lambda_1_min_idx, gamma_min_idx)= np.unravel_index(total_err.argmin(), total_err.shape)
    best_lambda_1 = lambda_1[lambda_1_min_idx]
    best_gamma = gamma[gamma_min_idx]
    print("(best_lambda_1 = {},  best_gamma = {}), and equals total_err[best_lambda_1, best_gamma]= {}".format(best_lambda_1, best_gamma, total_err[lambda_1_min_idx, gamma_min_idx]))
    
    #%% Now we build the predictor
    K_learnt = generate_exp_kernel(x, x, best_gamma)
    alpha_learnt = solve_for_alpha(K_learnt, best_lambda_1, y)
    
    #%% using learnt hyperparams and alpha, eval on a range of points with unif. interval
    num_new_test_points = 200
    new_test_points = np.linspace(0, 1, num_new_test_points)
    fx_new_test_points = compute_fx(new_test_points)
    fxhat_new_test_points = np.empty(num_new_test_points)
    for i in range(num_new_test_points):
        fxhat_new_test_points[i] = np.inner(alpha_learnt.T, generate_exp_kernel(new_test_points[i], x, best_gamma))
    
    #%% final stuff (for part b)
    sns.set()
    
    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y, c='k')
    ax.plot(new_test_points, fx_new_test_points, c='b')
    ax.plot(new_test_points, fxhat_new_test_points, c = 'r')
    ax.set_xlim([0,1])
    ax.set_ylim([-1.5, 41.5])
    ax.legend(['true fit', 'learnt fit', 'data'])
    ax.set_title('Learning exp. kernel hyperparams. with ls loss fn and non-decr constr.')    
