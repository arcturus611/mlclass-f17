#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:22:19 2017
Homework 3: Problem 1, last part
@author: arcturus
"""
#%% import libraries
import numpy as np
from scipy import linalg as la 
import matplotlib.pyplot as plt
import seaborn as sns

#%% We want n random vars x_i's such that x_i \sim N(0, I_2). this is equivalent to generating 2 iid x_ij's, each \sim N(0, 1)
def generate2dnormals(n):
    X = np.random.randn(2, n)
    return X

#%% compute the parameters we need from generated data
def computeParams(X):
    sumX = X.sum(axis=1, keepdims = True) #without the keepdims, default is to transpose the thing, which i don't like
    sumOuterProdX = X.dot(X.transpose())
    return sumX, sumOuterProdX

#%% based on calcs in homework (theory), we compute A and b
def computeAandb(V, Lam, mu_, cov_, n):
    # mu_ = \sum_i x_i and cov_ = \sum_i x_i x_i^T
    I_ = cov_ + 3*mu_.dot(mu_.transpose())/n
    A = V.dot(la.sqrtm(Lam)).dot(la.inv( la.sqrtm(I_) ))
    b = -A.dot(mu_)/n #ahh had forgotten minus sign earlier! now it's fixed!
    return A, b

#%% generate the z's
def generateZ(A, b, X, n):
    Z = A.dot(X) + np.matlib.repmat(b, 1, n)
    return Z

#%% plot the zis and conf ellipsoid
def plotZandEll(Z, V, Lam, w_star, fig, i):
    ax = fig.add_subplot(2, 3, i)
    ax.axis('equal')
    ax.scatter([Z[0, :]], [Z[1, :]], marker="o")
    A = V.dot(la.inv(la.sqrtm(Lam)))
    num_plot_points = 100
    x_coords = np.cos(np.linspace(0, 2*np.pi, num_plot_points))
    y_coords = np.sin(np.linspace(0, 2*np.pi, num_plot_points))
    coords = np.stack((x_coords, y_coords))
    tx_coords = A.dot(coords)
    ax.plot(w_star[0] + tx_coords[0, :], w_star[1] + tx_coords[1,:], 'r', linewidth=2) #see BV page 30: \Ell = {x_c + Au: u \in B_2(1)}
    ax.set_title("lam=({}, {}) and w*=({}, {})".format(Lam[0, 0], Lam[1, 1], w_star[0], w_star[1]))
    
        
#%% 
def main():
    #%% input data 
    V = np.array([[3/5, -4/5], [4/5, 3/5]])
#    w_star = np.array([3, 4])
#    (lam1, lam2) = (6, 1)
    w_star_mat = np.array([[3, 4], [3, 4], [3, 4], [-4, 3], [0, 4]])
    lam_mat = np.array([[1, 1], [8, 1], [1, 8], [8, 1], [8, 1]])
    #Lam = np.diag([lam1, lam2])
    n = 100
    #fig, ax = plt.subplots(2, 3)
    #ax = ax.ravel()
    fig = plt.figure()
    sns.set()
    for i in range(5):
        #%% generate X and its functions we need
        X = generate2dnormals(n)
        (mu_, cov_) = computeParams(X)
        Lam = np.diag(lam_mat[i, :])
        (A, b) = computeAandb(V, Lam, mu_, cov_, n)
        Z = generateZ(A, b, X, n)
        
        #%% print a bunch of stuff
        print("sum of xis is {} and sum of outer prods is {}".format(mu_, cov_))
        print("Value of V Lam Vt is {}".format(V.dot(Lam).dot(V.transpose())))
        print("sum of zis is {}".format(Z.sum(axis = 1, keepdims = True))) # this should be zero
        print("cov of zis is {}".format(Z.dot(Z.transpose())))
        
        #%% plot stuff
        w_star = w_star_mat[i, :]
        plotZandEll(Z, V, Lam, w_star, fig, i+1)
        fig.suptitle("Data with Confidence Ellipsoids")
#%% 
if __name__ == '__main__':
    main()    