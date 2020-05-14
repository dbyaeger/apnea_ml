#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:45:11 2020

@author: danielyaeger
"""
import numpy as np
import more_itertools
from pathlib import Path
import pickle
import pyximport
pyximport.install()
from viterbi.viterbi import post_process

def smooth_all_with_viterbi(path_to_probabilites: str,
                            save_path: str,
                            data_path: str,
                            with_targets: bool = True,
                            probabilities_file_name: str = 'five_conv_two_dense_test_set_results.p',
                            save_name: str = 'five_conv_two_dense_test_viterbi_smoothed.p',
                            transition_matrix_name: str = 'apnea_hypopnea_transition_matrix',
                            stage_file: str = 'stage_dict.p',
                            REM_only = False,
                            sampling_rate: int = 10) -> dict:
    """Smoothes probabilities for all IDs in dictionary of probabilities keyed by
    ID stored in probabilities_file_name in path_to_probabilites. Uses Viterbi
    alogrithm to maximize the number of correct positions. Generates output 
    save_name in save_path which is pickled dictionary keyed by ID with
    predictions and targets.
    
    INPUTS:
        path_to_probabilites: where probabilities dict lives
        save_path: where to save output
        data_path: where .npy data files and metadata lives
        with_targets: whether to include the targets in the output file.
        probabilities_file_name: name of probabilities dict
        save_name: name to give output file
        transition_matrix_name: name of transition matrix file, assumed to live in data_path
        stage_file: name of stage dictionary, assumed to live in data_path
        REM_only: whether apnea was only scored for REM stages
    OUTPUTS:
        pickled dictionary keyed by ID with predictions with predictions and 
        targets (if with_targets set to True) saved with name save_name in
        save_path.
    RETURNS: None
    """
    if not isinstance(path_to_probabilites, Path):
        path_to_probabilites = Path(path_to_probabilites)
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    
    with path_to_probabilites.joinpath(probabilities_file_name).open('rb') as fp:
        probabilities_dict = pickle.load(fp)
    
    viterbi_smoothed_preds = {}
    
    for ID in probabilities_dict:
        print(f'ID: {ID}')
        if with_targets:
            full_predictions, targets = smooth_with_viterbi(probabilites = probabilities_dict[ID]['predictions'],
                                                    ID = ID,
                                                    data_path = data_path,
                                                    return_targets = with_targets,
                                                    transition_matrix_name = transition_matrix_name,
                                                    stage_file = stage_file,
                                                    REM_only = REM_only,
                                                    sampling_rate = sampling_rate)
            viterbi_smoothed_preds[ID] = {'predictions': full_predictions,
                                          'targets': targets}
        else:
            full_predictions = smooth_with_viterbi(probabilites = probabilities_dict[ID]['predictions'],
                                                    ID = ID,
                                                    data_path = data_path,
                                                    return_targets = with_targets,
                                                    transition_matrix_name = transition_matrix_name,
                                                    stage_file = stage_file,
                                                    REM_only = REM_only,
                                                    sampling_rate = sampling_rate)
            viterbi_smoothed_preds[ID] = {'predictions': full_predictions}
        
        with save_path.joinpath(save_name).open('wb') as fout:
            pickle.dump(viterbi_smoothed_preds, fout)
            
        

def smooth_with_viterbi(probabilites: np.ndarray,
                        ID: str,
                        data_path: str,
                        return_targets: bool = True,
                        transition_matrix_name: str = 'apnea_hypopnea_transition_matrix',
                        stage_file: str = 'stage_dict.p',
                        REM_only = False,
                        sampling_rate: int = 10) -> np.ndarray:
    """ Smoothes probabilites and generates predictions using the Viterbi algorithm.
    INPUTS:
        probabilites: probabilistic predictions for A and H events for sleeper (1-D np.ndarray)
        ID: ID of sleeper
        data_path: where .npy data and metadata files live
        transition_matrix_name: name of transition matrix file
        return_targets: whether to return the full-length targets as well as
        the full-length predictions
        stage_file: name of stage_file, which lives in data_path
        REM_only: whether apnea is only considered during REM stages (as opposed to all sleep stages)
        sampling_rate: sampling rate of data (Hz)
    RETURNS:
        full_predictions: array of predictions where all parts of sleep are 
        included (i.e. wake, non-REM, and REM). A 0 may mean either no apnea
        or waking stage, whereas a 1 indicates both that subject was sleeping
        and apnea was detected. Predictions generated by smoothing probabilistic
        predictions using Viterbi search.
    
    """
    if type(data_path) == str: data_path = Path(data_path)
     
    with data_path.joinpath(stage_file).open('rb') as fs:
        stage_dict = pickle.load(fs)[ID]
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
        targets = pickle.load(ta)[ID]
    
    with data_path.joinpath(transition_matrix_name).open('rb') as tm:
        transition_mat = pickle.load(tm)
    
    filtered_target = targets.copy()
    for epoch in stage_dict:
        if REM_only:
            if stage_dict[epoch] not in ['R']:
                filtered_target[(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1
        elif not REM_only:
            if stage_dict[epoch] not in ['R','1','2','3']:
                filtered_target[(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1
    
    # Set up arrays to store full probabilities and predictions
    full_probabilities = np.zeros((len(targets),probabilites.shape[-1]))
    full_predictions = np.zeros(len(targets))
    
    # Convert to full length
    counter = 0
    for i in range(len(targets)):
        if filtered_target[i] != -1:
            full_probabilities[i,:] = probabilites[counter,:]
            counter += 1
            if counter >= len(probabilites): break
    
    if smooth_with_viterbi:
        # Get subsequence indices
        idx = np.nonzero(filtered_target != -1)[0]
        groups = [list(group) for group in more_itertools.consecutive_groups(idx)]
        
        # Get Viterbi smoothed predictions for each subsequence
        for group in groups:
            full_predictions[group] = post_process(full_probabilities[group,:], transition_mat)
    
    if return_targets:
        return full_predictions, targets
    else:
        return full_predictions
