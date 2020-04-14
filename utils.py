#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:12:57 2019

@author: macbookthibaultlahire
"""

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib.patches import Ellipse

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def plot_ellipse(mean, cov, ax, factor=np.sqrt(5.991), alpha=.2):
    # factor=np.sqrt(5.991) to print a 95% confidence interval 
    width = 2*np.sqrt((cov[0,0] + cov[1,1])/2 + np.sqrt(((cov[0,0]-cov[1,1])/2)**2 + cov[0,1]**2))
    height = 2*np.sqrt((cov[0,0] + cov[1,1])/2 - np.sqrt(((cov[0,0]-cov[1,1])/2)**2 + cov[0,1]**2))
    if cov[0,1] < 10**(-50) and cov[0,0] >= cov[1,1]:
        theta = 0
    elif cov[0,1] < 10**(-50) and cov[0,0] < cov[1,1]:
        theta = 90
    else:
        theta = np.arctan2(width - cov[0,0], cov[0,1])
    ellipse = Ellipse(xy=mean, width=factor*width, height=factor*height, angle=theta, edgecolor='k', linewidth=1., alpha=alpha)
    ax.add_patch(ellipse)
        


def gaussian_diag(X, num_classes, maxiter=100):
    n_samples = X.shape[0]
    dim = X.shape[1]
    # Initialization with KMeans
    km = KMeans(n_clusters=num_classes)
    km.fit(X)
    labels_kmean = km.predict(X)
    mus = km.cluster_centers_
    
    P = np.empty((num_classes, 1))
    for k in range(num_classes):
        P[k, 0] = np.sum(labels_kmean == k) / n_samples

    sigmas = np.stack([np.ones(dim) for k in range(num_classes)])

    q = np.empty((num_classes, n_samples))
    
    for t in range(maxiter):
        # E-STEP
        for k in range(num_classes):
            for i in range(n_samples):
                q[k,i] = P[k,0] * multivariate_normal.pdf(X[i], mean=mus[k], cov=np.diagflat(sigmas[k]))
        likelihood_array = np.zeros(n_samples)

        for i in range(n_samples):
            for k in range(num_classes):
                likelihood_array[i] += P[k,0] * multivariate_normal.pdf(X[i], mean=mus[k], cov=np.diagflat(sigmas[k]))
        for k in range(num_classes):
            for i in range(n_samples):
                q[k,i] = q[k,i] / likelihood_array[i]
        
        w = np.sum(q, axis = 1, keepdims=True)
        # M-STEP
        P[:, 0] = np.mean(q, axis = 1)
        
        for k in range(num_classes):

            mus[k] = 0
            for i in range(n_samples):
                mus[k] += q[k,i] * X[i]
            mus[k] = mus[k] / w[k]
            
            sig = np.zeros(dim)
            for l in range(dim):
                for i in range(n_samples):
                    sig[l] += q[k,i] * (X[i][l] - mus[k][l]) ** 2
                sig[l] = sig[l]/w[k]
            sigmas[k] = sig
        
    labels_diag = np.argmax(q, axis=0)
    likelihood = np.prod(likelihood_array)
    
    return P, mus, sigmas, labels_diag, likelihood


if __name__ == '__main__':
    num_classes = 3
    iris_data = load_iris()
    X = iris_data.data
    X_T = X.T
    labels_real = iris_data.target
    P, mus, sigmas, _, __ = gaussian_diag(X, 3)
    print(P)
    print(mus)
    print(sigmas)

