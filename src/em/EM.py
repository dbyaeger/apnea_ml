#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:35:03 2020

@author: danielyaeger
"""
import numpy as np
from scipy.special import logsumexp

def estep(train_posterior: np.ndarray, prior: np.ndarray,
          train_prior: np.ndarray, epsilon: float = 1e-16) -> (np.ndarray, float):
    """Calculates posteriors P(class | X) for test
    data.

    Args:
        train_posterior: (n, k) array holding the posteriors calculated during
            training
        prior: (k,) array holding priors
        train_prior: (k,) array holding training priors
        epsilon: float to prevent numerical underflow

    Returns:
        posteriors: (n, k) array holding the posteriors for the new data
        float: log-likelihood of the assignment
    """

    post = np.zeros(train_posterior.shape)
    
    ll = 0
    
    for i in range(train_posterior.shape[0]):
        for j in range(len(prior)):
            
            numerator = np.log(prior[j] + epsilon) + \
            np.log(train_posterior[i,j] + epsilon) - \
            np.log(train_prior[j] + epsilon)
            
            post[i,j] = numerator
        
        total = logsumexp(post[i,:])
        post[i, :] = post[i, :] - total
        ll += total
    
    return np.exp(post), ll


def mstep(posteriors: np.ndarray) -> np.ndarray:
    """M-step: Updates the priors for the new data by maximizing the log-likelihood
    of the posteriors.

    Args:
        posteriors: (n, d) array holding the data
    Returns:
        priors: (k,) the new priors
    """
    N = posteriors.shape[0]
    return posteriors.sum(axis=0)/N


def run(train_posterior: np.ndarray, train_prior: np.ndarray) -> (np.ndarray, np.ndarray):
    """Implements the EM algorithm in "Adjusting the outputs of a classifier to
    new a priori probabilities: A simple procedure" by Saerens, Latinne, and 
    Decaestecker. Returns the posteriors and priors which maximize the log
    likelihood for the new data, based on the posteriors and priors obtained
    during training.

    Args:
        train_posterior: (n, k) array holding the posteriors calculated during
            training
        train_prior: (k,) array holding array holding training priors

    Returns:
        new_posterior: (n,k) array holding the updated posteriors
        new_priors: (k,) array holding the updated priors
        
    """
    prev_cost = None
    # Initially set priors to train_priors
    posteriors, cost = estep(train_posterior = train_posterior, prior = train_prior,
          train_prior = train_prior)
    while (prev_cost is None or cost - prev_cost > (1e-6)*np.abs(cost)):
        prev_cost = cost
        priors = mstep(posteriors)
        posteriors, cost = estep(train_posterior = train_posterior, prior = priors,
          train_prior = train_prior)
    return posteriors, priors
