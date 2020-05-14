#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:53:21 2020

@author: danielyaeger
"""
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from collections import Counter
import numpy as np
from em.EM import run, estep

def test_em(RANDOM_STATE: int = 0):
    """Creates an artificial unbalanced data set. Trains a logistic regression
    classifier on a balanced version of the data. Then evaluates the classifier
    on held-out data, and also corrects these posterior probabilities using
    the EM algorithm. Finally prints out the classification accuracy without
    correcting the prior probabilities, with correcting using the EM algorithm,
    and with correcting using the true prior probabilities.
    """
    X, y = make_classification(n_samples = 5000, n_features = 2,
                               n_informative = 2, n_redundant = 0,
                               n_repeated = 0, n_classes = 2,
                               n_clusters_per_class = 1,
                               weights = [1/3,2/3],
                               class_sep = 0.1,
                               random_state = RANDOM_STATE)
    
    # Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        shuffle = True,
                                                        random_state=RANDOM_STATE)
    
    # Calculate prior probabilities for test set
    counts = Counter(y_test)
    true_test_prior = np.array([counts[key] for key in sorted(list(counts.keys()))])
    true_test_prior = true_test_prior/true_test_prior.sum()
    
    # Resample
    rus = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy = 1)
    X_train_resample, y_train_resample = rus.fit_resample(X_train, y_train)
    
    # Calculate train_prior
    counts = Counter(y_train_resample)
    train_prior = np.array([counts[key] for key in sorted(list(counts.keys()))])
    train_prior = train_prior/train_prior.sum()
    
    # Train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(X_train_resample, y_train_resample)
    
    # Get posterior probabilites on test set
    y_pred = clf.predict_proba(X_test)
    
    # Estimate prior probabilites with EM
    est_posteriors, est_priors = run(train_posterior = y_pred,
                                     train_prior = train_prior)
    
    # Manually fix posterior
    true_posteriors, _ = estep(train_posterior = y_pred, 
                              prior = true_test_prior,
                              train_prior = train_prior)
    
    print(f'True priors: {true_test_prior}')
    print(f'Estimated priors: {est_priors}\n\n')
    print(f'Accuracy with no adjustment: {accuracy_score(y_test,y_pred.argmax(-1))}')
    print(f'Balanced accuracy with no adjustment: {balanced_accuracy_score(y_test,y_pred.argmax(-1))}\n')
    print(f'Accuracy with EM adjustment: {accuracy_score(y_test,est_posteriors.argmax(-1))}')
    print(f'Balanced accuracy with EM adjustment: {balanced_accuracy_score(y_test,est_posteriors.argmax(-1))}\n')
    print(f'Accuracy with true priors: {accuracy_score(y_test,true_posteriors.argmax(-1))}')
    print(f'Balanced accuracy with true priors: {balanced_accuracy_score(y_test,true_posteriors.argmax(-1))}')
    

if __name__ == "__main__":
    test_em()