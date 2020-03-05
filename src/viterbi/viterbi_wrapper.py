#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:45:11 2020

@author: danielyaeger
"""
import numpy as np
import pyximport
pyximport.install()
from viterbi import post_process

def viterbi_wrapper(transition_mat: np.array, data: dict) -> dict:

    """ Takes a transition matrix and dictionary keyed by ID with predicted probabilites
    for each class.
    
    INPUTS:
        transition matrix: numpy array with probabilites for transitioning from
        state in column to state in row.
        
        data: dictionary keyed by ID, with prediction matrix as value for each ID.
        
            e.g. data[ID] = array([[0.9., 0.1, 0.],
                                  [0.3, 0.2, 0.5],
                                  [0.1, 0.3, 0.6]])
    RETURNS
        smooth_output: dictionary keyed by ID, with smoothed output lables as value
        for each ID: e.g.
            smooth_output[ID] = np.array([0,0,1,1......])
        
    """
    

    smooth_output = {}
        
    for ID in data:
        print(f'Smoothing {ID}')
        smooth_output[ID] = post_process(data[ID], transition_mat)

    return smooth_output