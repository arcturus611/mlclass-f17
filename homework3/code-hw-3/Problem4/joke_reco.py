#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:57:32 2017

@author: arcturus
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import fileinput
from scipy.sparse.linalg import svds, eigs
import seaborn as sns

#%% estimator for part b (compute SVDs)
def est_svds(ratings, d):
    # Input: ratings matrix (n users X m jokes), d = # eigvals you want
    # Output: U, S, V, per std notation , such that ratings \approx \sum_{i = 1}^d S[i]*np.outer(U[:, i], V[i, :]). 
    # NOTE: S is an array, not a diag matrix
    
    U, S, V = svds(ratings, k = d)
    return U, S, V
    
#%% predictor for part b 
def pred_from_svds(U, V):
    # Input: U, V
    # Output: predicted matrix computed by unweighted outer prods
    # WARNING: VERY LIKELY INCORRECT 
    pred = U.dot(V)
    return pred

#%% 
def compute_err_basic(filename, basic_pred, num_ratings_per_joke):    
    #%% 
    f = open(filename, "r")
    squared_error = 0
    abs_error= 0
    
    #%%
    while True:
        line = f.readline()
        if(line==""):
            break
        
        data = line.split(",")
                        
        squared_error+= np.square(float(data[2]) - basic_pred[int(data[1]) - 1])
        
        abs_error+= np.abs(float(data[2]) - basic_pred[int(data[1]) - 1])
        
    #%%        
    f.close()
    mean_squared_error = squared_error/np.sum(num_ratings_per_joke)
    mean_abs_error = abs_error/np.sum(num_ratings_per_joke)
    return mean_squared_error, mean_abs_error

#%% 
def compute_err_svd(filename, ratings_hat, num_ratings_per_joke):    
    #%% 
    f = open(filename, "r")
    squared_error = 0
    abs_error= 0
    
    #%%
    while True:
        line = f.readline()
        if(line==""):
            break
        
        data = line.split(",")
                        
        squared_error+= np.square(float(data[2]) - ratings_hat[int(data[0])-1, int(data[1]) - 1])
        
        abs_error+= np.abs(float(data[2]) - ratings_hat[int(data[0])-1, int(data[1]) - 1])
        
    #%%        
    f.close()
    mean_squared_error = squared_error/np.sum(num_ratings_per_joke)
    mean_abs_error = abs_error/np.sum(num_ratings_per_joke)
    return mean_squared_error, mean_abs_error


#%%
def create_datastr(filename, num_users, num_jokes):
    #%% 
    f = open(filename, "r")
    ratings = np.zeros((num_users, num_jokes))
    total_val_per_joke = np.zeros(num_jokes)
    num_ratings_per_joke = np.zeros(num_jokes)
    
    #%%
    while True:
        line = f.readline()
        if(line==""):
            break
        
        data = line.split(",")
                
        ratings[int(data[0]) - 1, int(data[1]) - 1] = float(data[2])
        
        total_val_per_joke[int(data[1]) - 1]+= float(data[2])
        
        num_ratings_per_joke[int(data[1]) - 1]+= 1
    
    #%%      
    f.close() 
    return ratings, total_val_per_joke, num_ratings_per_joke

#%% estimator for part a (simply predict mean rating)
def est_trivial(ratings):
    # Input: raings matrix (n users X m jokes)
    # Output: predicted ratings for all m jokes (so a 1 X m vector)
    
    pred = np.mean(ratings, axis = 0)
    return pred

#%% 
if __name__ == '__main__':
    #%% 
    num_users = 24983
    num_jokes = 100
    total_val_per_joke = np.empty(num_jokes)
    num_ratings_per_joke = np.empty(num_jokes)
    
    #%%
    [ratings_train, total_val_per_joke, num_ratings_per_joke] = create_datastr("train.txt", num_users, num_jokes)

    #%% 
    basic_pred = total_val_per_joke/num_ratings_per_joke #this is our dumb predictor
        
    #%% 
    train_mse_basic, train_mae_basic = compute_err_basic("train.txt", basic_pred, num_ratings_per_joke)    
    test_mse_basic, test_mae_basic = compute_err_basic("test.txt", basic_pred, num_ratings_per_joke)
    
    #%% 
    all_degrees = np.array([1, 2, 5, 10, 20])
    (train_mse_svd, train_mae_svd, test_mse_svd, test_mae_svd) = (np.empty(5), np.empty(5), np.empty(5), np.empty(5))
    #%%
    for d in np.arange(len(all_degrees)):
        U, S, V = est_svds(ratings_train, all_degrees[d])
        ratings_hat = pred_from_svds(U, V)
        train_mse_svd[d], train_mae_svd[d] = compute_err_svd("train.txt", ratings_hat, num_ratings_per_joke)
        test_mse_svd[d], test_mae_svd[d] = compute_err_svd("test.txt", ratings_hat, num_ratings_per_joke)

    #%%
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax1.plot(all_degrees, train_mse_svd, 'r-')
    ax1.set_title('training mse')
    
    ax2 = fig.add_subplot(222)
    ax2.plot(all_degrees, test_mse_svd, 'b-')
    ax2.set_title('test mse')
    
    ax3 = fig.add_subplot(223)
    ax3.plot(all_degrees, train_mae_svd, 'r-.')
    ax3.set_title('training mae')
    
    ax4 = fig.add_subplot(224)
    ax4.plot(all_degrees, test_mae_svd, 'b-.')
    ax4.set_title('test mae')
    
    fig.suptitle('SVD Method (error vs degrees)')

    #%%
    
    