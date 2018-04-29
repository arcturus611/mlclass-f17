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
import scipy as sp
from scipy import stats
import seaborn as sns
#%%
def load_dataset(): # this function taken from the assignment verbatim
    mndata = MNIST('../../Homework1/hw1-problem-7/python-mnist-0.3/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

#%% 
def init_centers_rand(k, data):
    #Input: Number of clusters (k), data (n X d)
    #Output: centroids, randomly picked
    (n, d) = data.shape
    indices = np.random.randint(low = 0, high = n, size = k)
    centroids = data[indices, :]
    return centroids

#%% data in the form n X d, n = # data points, d = # dimensions
def init_centers_kmeanspp(k, data):
    # Input: Number of clusters (k), dta to be clustered (num_points X num_dimensions)
    # Output: Centers of the points (k X d)
    
    # Initialize
    (n, d) = data.shape
    centroids = np.empty(shape=(k, d))
    # first one is at random
    # assign all-zeros k-len col vector
    mu = np.empty(k, dtype= int) 
    mu[0] = np.random.randint(low = 0, high = k+1) #since randint returns [low, high)
    centroids[0, :] = data[mu[0], :] #copy mu[0]'th row as 0th centroid
    
    # compute distances of each data point from the picked initial centroid
    distances = np.sum((data - centroids[0, :])**2, axis = 1, keepdims = True) #keepdims ensures you get a col vec back. 
    
    # create distriubution and sample from it 
    indices = np.arange(n)
    centers_dist = stats.rv_discrete(values = (indices, distances/np.sum(distances)))
    mu[1] = centers_dist.rvs()
    centroids[1, :] = data[mu[1], :]
    
    # suppoes k = 5 (want to go till 4th otained)
    for i in range(2, k):
        # compute distances
        new_distances = np.sum((data - centroids[i-1, :])**2, axis = 1, keepdims = True)
        new_distances = np.minimum(new_distances, distances) #compare with prev best
        
        # sample acc to distances
        centers_dist = stats.rv_discrete(values = (indices, new_distances/np.sum(new_distances)))
        mu[i] = centers_dist.rvs() #sample from new dist
        print("new {}-th center is the {}-th data point".format(i, mu[i]))
        # find best centroid using this sampling and reset "distances"
        centroids[i, :] = data[mu[i], :]
        distances = new_distances
        
    return centroids

#%% 
def assign_to_centroids(data, centers):
    # assign a label to each data point, saying which 'group' it belongs to
    # Input: data (n X d), centers (k X d)
    # Output: cluster_labels (1 X n)
    
    # To do this, compute a n X k matrix (for each of n data points, compute k distances)
    (n, d) = data.shape
    (k, d) = centers.shape
    print("k = {} and n ={}".format(k, n))
    
    # beautiful speed-up using this instead of stupid for loop       
    dist_mat = sp.spatial.distance.cdist(centers, data, metric='euclidean')
    # Then, compute the min for each data point (and return an n X 1 vector)
    cluster_labels = np.argmin(dist_mat, axis = 0)
    
    # note, cluster_label(i) tells you which of [1, k] centroids the point i belongs to. 
    # for re-centering, these centroids don't matter
    # all that matters is, we know which data points are in one cluster
    print("assigned each point to a cluster")
    return cluster_labels

#%% 
def re_center(data, cluster_labels, num_clusters):
    # input 
    (n, d) = data.shape
    k = num_clusters
    
    # initialize
    centroids = np.empty(shape = (k, d))
    cluster_error = np.empty(k)
    
    # for ecah cluster, compute centroid
    for i in range(k):
        # index into those rows that belong to cluster i
        cluster_points = data[cluster_labels==i, :]
        
        # take the mean of the selected points and compute the centroid
        centroids[i,:] = np.mean(cluster_points, axis = 0, keepdims = True)
        
        # compute the sum of squared-distances from the centroid, of points in a cluster
        cluster_error[i] = np.sum( np.sum((cluster_points - centroids[i,:])**2, axis = 1) )
    
    # errors (for plots)
    total_cluster_error = np.sum(cluster_error)
    print("re-centering done!")
    return centroids, total_cluster_error
#%% 
if __name__ == '__main__':
    num_clusters = np.array([5, 10, 20]) #number of clusters (chosen arbitrarily for now)
    num_iters = 30
    [X_train, labels_train, X_test, labels_test] = load_dataset()

    total_error = np.empty(shape = (num_clusters.size, num_iters))    
    for k in range(num_clusters.size):
        #%%
        centroids = init_centers_rand(num_clusters[k], X_train)
        #centroids = init_centers_kmeanspp(num_clusters[k], X_train)# UNCOMMENT FOR KMEANS++ INITIALIZATION
        print("Centers initialized!")
    
        #%%
        for i in range(num_iters):
            cluster_labels = assign_to_centroids(X_train, centroids)
            [centroids, total_error[k, i]] = re_center(X_train, cluster_labels, num_clusters[k])


    #%% plot things
    sns.set()
    plt.figure(1)
    xvals = list(range(num_iters))
    plt.plot(xvals, total_error[0, :], 'r<', xvals, total_error[1, :], 'gs', xvals, total_error[2, :], 'bo')
    plt.xlabel('number of iterations')
    plt.ylabel('total squared errors from resp. centroids')
    plt.legend(['Five clusters', 'Ten clusters', 'Twenty clusters'])
    plt.title('Lloyds Alg with random init')

    fig2, ax2 = plt.subplots(5, 4)
    for i in range(5):
        for j in range(4):
            ax2[i, j].imshow(centroids[i*4+j, :].reshape((28, 28)))
    plt.suptitle('All 20 cluster centers')
    plt.show()    
