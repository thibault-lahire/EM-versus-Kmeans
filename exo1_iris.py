#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:54:11 2019

@author: macbookthibaultlahire
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from utils import gaussian_diag
from utils import plot_ellipse
from utils import purity_score


num_classes = 3

# Load data
iris_data = load_iris()
X = iris_data.data
X_T = X.T
labels_real = iris_data.target

# K-means:
km = KMeans(n_clusters=num_classes)
km.fit(X)
labels_kmeans = km.predict(X)
centroids = km.cluster_centers_

# GM with full matrix
gm = GaussianMixture(n_components=num_classes)
gm.fit(X)
labels_full = gm.predict(X)
mus_full = gm.means_
sigmas_full = gm.covariances_

# GM with diagonal matrix
P, mus_diag, sigmas_vect, labels_diag, likelihood = gaussian_diag(X, num_classes)
aic_diag = 2*(num_classes*(2*X.shape[1]+1) - np.log(likelihood))

sigmas_diag = []
for i in range(sigmas_vect.shape[0]):
    sigmas_diag.append(np.diag(sigmas_vect[i]))
sigmas_diag = np.array(sigmas_diag)




print("K means :")
purity_k_means = purity_score(labels_real, labels_kmeans)
print("PURITY =", purity_k_means)
print()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

try:
    axes = axes.ravel()
except AttributeError:
    axes = [axes]

count = 0
for i in range(X.shape[1]-1):
    for j in range(i+1,X.shape[1]):
        feat = str([i,j])
        X_plt = np.stack((X.T[i],X.T[j]))
        ax = axes[count]
        colors = ["lightblue", "lightgreen", "pink", "purple"]
        for k in range(num_classes):
            numclas = "Class " + str(k)
            ax.scatter(X_plt[0][labels_kmeans == k], X_plt[1][labels_kmeans == k], color=colors[k], label=numclas)
        ax.legend()
        k_mean_plt = np.stack((centroids.T[i], centroids.T[j]))
        ax.scatter(k_mean_plt[0],k_mean_plt[1],s=200,marker='+', color="black")
        title = 'features ' + feat + ', K-means with ' + str(int(num_classes)) + ' classes'
        ax.set_title(title)
        count +=1
fig.tight_layout()




print("GM with full matrix :")
purity_full = purity_score(labels_real, labels_full)
print("PURITY =", purity_full)
print("AIC =", gm.aic(X))
print()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

try:
    axes = axes.ravel()
except AttributeError:
    axes = [axes]

count = 0
for i in range(X.shape[1]-1):
    for j in range(i+1,X.shape[1]):
        feat = str([i,j])
        X_plt = np.stack((X.T[i],X.T[j]))
        ax = axes[count]
        mus_plt = np.stack((mus_full.T[i], mus_full.T[j]))
        colors = ["lightblue", "lightgreen", "pink", "purple"]
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
        title = 'features ' + feat + ', full gaussian with ' + str(int(num_classes)) + ' classes'
        ax.set_title(title)
        count +=1
fig.tight_layout()




print("GM with diagonal matrix :")
purity_diag = purity_score(labels_real, labels_diag)
print("PURITY =", purity_diag)
print("AIC =", aic_diag)
# Just to have a comparison with the method already implemented:
gm2 = GaussianMixture(n_components=num_classes, covariance_type='diag')
gm2.fit(X)
print("Comparison with the AIC of the GM diag already implemented in sklearn :",gm2.aic(X))
print()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

try:
    axes = axes.ravel()
except AttributeError:
    axes = [axes]

count = 0
for i in range(X.shape[1]-1):
    for j in range(i+1,X.shape[1]):
        feat = str([i,j])
        X_plt = np.stack((X.T[i],X.T[j]))
        ax = axes[count]
        mus_plt = np.stack((mus_diag.T[i], mus_diag.T[j]))
        colors = ["lightblue", "lightgreen", "pink", "purple"]
        for k in range(num_classes):
            numclas = "Class " + str(k)
            ax.scatter(X_plt[0][labels_diag == k], X_plt[1][labels_diag == k], color=colors[k], label=numclas)
            cov_mat_k = sigmas_diag[k]
            cov_mat_k = np.stack((cov_mat_k[i], cov_mat_k[j]))
            cov_mat_k = cov_mat_k.T
            cov_mat_k = np.stack((cov_mat_k[i], cov_mat_k[j]))
            plot_ellipse(mean=mus_plt.T[k],cov=cov_mat_k,ax=ax)
        ax.legend()
        ax.scatter(mus_plt[0],mus_plt[1],s=200,marker='+', color="black")
        title = 'features ' + feat + ', diag gaussian with ' + str(int(num_classes)) + ' classes'
        ax.set_title(title)
        count +=1
fig.tight_layout()

