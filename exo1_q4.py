#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:09:41 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from utils import plot_ellipse


np.random.seed(1)


num_classes = 3
n_samples = 1000

sig1 = np.array([[5000,0],[0,1]])
sig2 = np.array([[1,0],[0,5000]])
mu1 = np.array([0,-20])
mu2 = np.array([100,20])
mu3 = np.array([-100,15])

X1 = np.random.multivariate_normal(mu1,sig1,n_samples)
Y1 = np.array([0 for i in range(n_samples)])
X2 = np.random.multivariate_normal(mu2,sig1,n_samples)
Y2 = np.array([1 for i in range(n_samples)])
X3 = np.random.multivariate_normal(mu3,sig2,n_samples)
Y3 = np.array([2 for i in range(n_samples)])

X = np.concatenate((X1,X2,X3))
labels_real = np.concatenate((Y1,Y2,Y3))


# K-means
km = KMeans(n_clusters=num_classes)
km.fit(X)
labels_kmean = km.predict(X)
centroids = km.cluster_centers_

# GM with full matrix
gm = GaussianMixture(n_components=num_classes)
gm.fit(X)
labels_full = gm.predict(X)
mus_full = gm.means_
sigmas_full = gm.covariances_



# Plot the results


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

try:
    axes = axes.ravel()
except AttributeError:
    axes = [axes]

count = 0
for i in range(X.shape[1]):
    for j in range(i+1,X.shape[1]):
        string = str([i,j])
        X_plt = np.stack((X.T[i],X.T[j]))
        ax = axes[count]
        colors = ["lightblue", "lightgreen", "pink"]
        for k in range(num_classes):
            numclas = "Class " + str(k)
            ax.scatter(X_plt[0][labels_kmean == k], X_plt[1][labels_kmean == k], color=colors[k], label=numclas)
        ax.legend()
        k_mean_plt = np.stack((centroids.T[i], centroids.T[j]))
        ax.scatter(k_mean_plt[0],k_mean_plt[1],s=200,marker='+', color="black")
        ax.set_title('K-means with K = %s' %num_classes)
        count +=1
fig.tight_layout()
fig.savefig('Kmeans.png')





fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

try:
    axes = axes.ravel()
except AttributeError:
    axes = [axes]

count = 0
for i in range(X.shape[1]):
    for j in range(i+1,X.shape[1]):
        string = str([i,j])
        X_plt = np.stack((X.T[i],X.T[j]))
        ax = axes[count]
        mus_plt = np.stack((mus_full.T[i], mus_full.T[j]))
        ax.set_title('features %s' %string)
        colors = ["lightblue", "lightgreen", "pink"]
        for k in range(num_classes):
            numclas = "Class " + str(k)
            ax.scatter(X_plt[0][labels_full == k], X_plt[1][labels_full == k], color=colors[k], label=numclas)
            cov_mat_k = sigmas_full[k]
            cov_mat_k = np.stack((cov_mat_k[i], cov_mat_k[j]))
            cov_mat_k = cov_mat_k.T
            cov_mat_k = np.stack((cov_mat_k[i], cov_mat_k[j]))
            plot_ellipse(mean=mus_plt.T[k],cov=cov_mat_k,ax=ax)
        ax.legend()
        ax.scatter(mus_plt[0],mus_plt[1],s=200,marker='+', color="black")
        ax.set_title('Gaussian (full) with K = %s' %num_classes)
        count +=1
fig.tight_layout()
fig.savefig('EM_GMM.png')
