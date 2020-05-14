#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:50:30 2020

@author: danielyaeger
"""
from pathlib import Path
import numpy as np
import pickle

def convert_from_samples_to_full_predictions(path_to_predictions: str,
                                             save_path: str,
                                             data_path: str,
                                             predictions_file_name: str = 'five_conv_two_dense_test_set_results.p',
                                             save_name: str = 'five_conv_two_dense_test_full_predictions_no_HMM.p',
                                             stage_file: str = 'stage_dict.p',
                                             predictions_only: bool = False,
                                             REM_only = False,
                                             sampling_rate: int = 10):
    """Generates a file with predictions for full sleep (i.e. all times during 
    the sleep study, including wake stages).
    INPUTS:
        path_to_predictions: where prediction file lives
        save_path: where to save full predictions
        data_path: where data and stage dictionary lives
        predictions_file_name: name of file containing predictions, keyed by ID
        save_name: what to call the file output from this function
        stage_file: name of stage dictionary
        predictions_only: if true, output only contain the full predictions for
            each ID. Otherwise, it contains full predictions for both target and 
            prediction.
        REM_only: whether apnea predictions were only made for REM stage
        sampling_rate: sampling_rate of data
    OUTPUTS:
        saves a dictionat with full predictions, keyed by ID, in the save_path
        with name save_name
    """
    if not isinstance(path_to_predictions, Path): path_to_predictions = Path(path_to_predictions)
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    with path_to_predictions.joinpath(predictions_file_name).open('rb') as fp:
        predictions = pickle.load(fp)
    
    full_predictions = {ID: {} for ID in predictions}
    for ID in predictions:
        print(f'ID: {ID}')
        full_predictions[ID]['predictions'] = \
            go_from_sample_to_full_predictions(predictions = predictions[ID]['predictions'],
                                               ID = ID,
                                               data_path = data_path,
                                               stage_file = stage_file,
                                               REM_only = REM_only,
                                               sampling_rate = sampling_rate)
        if not predictions_only:
            full_predictions[ID]['targets'] = \
            go_from_sample_to_full_predictions(predictions = predictions[ID]['targets'],
                                               ID = ID,
                                               data_path = data_path,
                                               stage_file = stage_file,
                                               REM_only = REM_only,
                                               sampling_rate = sampling_rate)
    
    with save_path.joinpath(save_name).open('wb') as fout:
        pickle.dump(full_predictions,fout)

def go_from_sample_to_full_predictions(predictions: np.ndarray,
                                       ID: str,
                                       data_path: str,
                                       stage_file: str = 'stage_dict.p',
                                       REM_only = False,
                                       sampling_rate: int = 10) -> np.ndarray:
    """ Solves the problem of generating labels for segments excluded from testing
    because predicted to be NREM/Wake or apneic/hypopneic for RSWA signal-level
    classification.
    INPUTS:
        ID: ID of sleeper
        data_path: where .npy data and metadata .p files live
        predictions: predictions for A and H events for sleeper (1-D np.ndarray)
        stage_file: name of stage_file, which lives in data_path
        REM_only: whether apnea is only considered during REM stages (as opposed to all sleep stages)
        sampling_rate: sampling rate of data (Hz)
    RETURNS:
        full_predictions: array of predictions where all parts of sleep are 
        included (i.e. wake, non-REM, and REM). A 0 may mean either no apnea
        or waking stage, whereas a 1 indicates both that subject was sleeping
        and apnea was detected.
    
    """
    if type(data_path) == str: data_path = Path(data_path)
     
    with data_path.joinpath(stage_file).open('rb') as fs:
            stage_dict = pickle.load(fs)[ID]
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
            targets = pickle.load(ta)[ID]
    
    filtered_target = targets.copy()
    for epoch in stage_dict:
        if REM_only:
            if stage_dict[epoch] not in ['R']:
                filtered_target[(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1
        elif not REM_only:
            if stage_dict[epoch] not in ['R','1','2','3']:
                filtered_target[(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1
    
    counter = 0
    full_predictions = np.zeros(len(targets))

    # Predictions may be a list
    if isinstance(predictions, list):
      predictions = np.array(predictions)
    
    # Predictions may be probabilites
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(-1)
    
    for i in range(len(targets)):
        if filtered_target[i] != -1:
            full_predictions[i] = predictions[counter]
            counter += 1
            if counter >= len(predictions): break
    
    assert len(full_predictions) >= len(predictions), f'Length of full_predictions must be greater than or equal to length of predictions!'
    
    return full_predictions

    
    