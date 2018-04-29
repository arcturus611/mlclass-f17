#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:00 2017
Code for Problem 1
@author: arcturus
"""

#%%
" libraries for math, plots "
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy as sp

#%% part a
" input data: number of points, gaussian parameters"
n = 100

mu1 = np.matrix('1;2')
cov1 = np.matrix('1 0; 0 2')

mu2 = np.matrix('-1;1')
cov2 = np.matrix('2 -1.8; -1.8 2')

mu3 = np.matrix('2; -2')
cov3 = np.matrix('3 1; 1 2')

#%% 
" given X \sim N(mu, sigma), Y = AX + B \sim N(A*mu + B, A*sigma*A^T) " 
X = np.random.randn(2, n)

#%% 
for i in range(0, 3):
    
    # how to stack 2d arrays in 3d to avoid such if-constructs?
    if i==0:
        mu = mu1
        cov = cov1
    elif i==1:
        mu = mu2
        cov = cov2
    else:
        mu = mu3
        cov = cov3

    A = sp.linalg.sqrtm(cov)
    
    Y = np.matlib.matmul(A, X) + np.matlib.repmat(mu, 1, n)
    #%% create a figure box
    plt.figure(i+1)

    " scatter plot everything "
    plt.scatter([Y[0, :]], [Y[1, :]], marker=">")

    #%% part b
    " empirical mean "
    mu_hat = Y.sum(axis = 1)/n
 
    #%%
    " empirical covariance "
    de_mean = Y - np.matlib.repmat(mu_hat, 1, n)
    cov_hat = np.matlib.matmul( de_mean , de_mean.transpose() )/(n-1)   
    
    #%%
    " eigenvalues of empirical cov "
    eigvals, eigvecs = LA.eig(cov_hat)

    plt.plot([mu_hat[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[0])*eigvecs[0, 0], mu_hat[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[1])*eigvecs[0, 1]], [mu_hat[1, 0], mu_hat[1]+np.sqrt(eigvals[0])*eigvecs[1, 0], mu_hat[1, 0], mu_hat[1]+np.sqrt(eigvals[1])*eigvecs[1, 1]], 'k')

    #%% 
    " whitening the data" 
    one_dim = (1/np.sqrt(eigvals[0]))*eigvecs[:,0].transpose()*de_mean
    second_dim = (1/np.sqrt(eigvals[1]))*eigvecs[:, 1].transpose()*de_mean
    
    plt.scatter([one_dim], [second_dim], marker="o")
    
    plt.title('Dataset ' + str(i+1) )
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.legend([ 'Scaled Eigenvecs', 'Orig Data', 'Whitened Data'])
    
    plt.savefig('Dataset' + str(i+1) + '.png', format='png')
    
    #%% stuff that didn't work
    #    plt.plot([mu_hat[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[1])*eigvecs[0, 1]], [mu_hat[1, 0], mu_hat[1]+np.sqrt(eigvals[1])*eigvecs[1, 1]], 'k')
#    X = [mu_hat[0, 0]+np.sqrt(eigvals[0])*eigvecs[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[1])*eigvecs[0, 1]]
#    Y = [mu_hat[1]+np.sqrt(eigvals[0])*eigvecs[1, 0], mu_hat[1]+np.sqrt(eigvals[1])*eigvecs[1, 1]]
#    U = [mu_hat[0, 0], mu_hat[0, 0]]
#    V = [mu_hat[1, 0], mu_hat[1, 0]]
##   plt.quiver([mu_hat[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[0])*eigvecs[0, 0], mu_hat[0, 0], mu_hat[0, 0]+np.sqrt(eigvals[1])*eigvecs[0, 1]], [mu_hat[1, 0], mu_hat[1]+np.sqrt(eigvals[0])*eigvecs[1, 0], mu_hat[1, 0], mu_hat[1]+np.sqrt(eigvals[1])*eigvecs[1, 1]])
#    plt.quiver(U, V, X, Y)