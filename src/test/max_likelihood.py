#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:35:53 2020

@author: danielyaeger
"""
import numpy as np
from pathlib import Path
import pickle

def maxlikelihood(path_to_probabilities: str,
                  save_path: str,
                  probablities_file_name: str = 'five_conv_two_dense_test_set_results.p',
                  save_name: str = 'five_conv_two_dense_test_max_likelihood.p',):
    
    if not isinstance(path_to_probabilities, Path): path_to_probabilities = Path(path_to_probabilities)
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    with path_to_probabilities.joinpath(probablities_file_name).open('rb') as fp:
        probabilities = pickle.load(fp)
    
    ml_predictions = {ID: {} for ID in probabilities}
    for ID in probabilities:
        ml_predictions[ID]['predictions'] =  probabilities[ID]['predictions'].argmax(1)
        targets = probabilities[ID]['targets']
        if isinstance(targets, list): targets = np.array(targets)
        ml_predictions[ID]['targets'] = targets
    
    with save_path.joinpath(save_name).open('wb') as fs:
        pickle.dump(ml_predictions,fs)