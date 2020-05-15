#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:38:11 2020

@author: danielyaeger
"""
import numpy as np
from pathlib import Path
import pickle

def get_epoch_level_predictions_for_evaluation(ID: str,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                data_path: str,
                                stage_file: str,
                                apnea_threshold_for_epoch: float, 
                                sampling_rate: int = 10, 
                                epoch_length: int = 30) -> (np.ndarray, np.ndarray):
    """Takes in ID and arrays of predictions and targets and returns
    signal-level predictions and targets and epoch-level predictions and targets
    NPUTS:
        ID: sleeper ID.
        data: dictionary with predictions and targets
        data_path: where stage file and apnea/hypopnea targets live
        stage_file: name of staging file
        save_path: where to save results
        save_name: what to name saved results
        apnea_threshold_for_epoch: the total length of apneic events required for a
            sleep epoch to be called an apnea epoch
        sampling_rate: sampling_rate at which labels were assigned, typically 10 Hz
        epoch_length: epoch_length in seconds, typically 30 s
    
    RETURNS:
        pandas dataframe showing results. Also saves results as a .csv file
        using the save_path and save_name input arguments.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    
    with data_path.joinpath(stage_file).open('rb') as fs:
        stage_dict = pickle.load(fs)[ID]
    
    if isinstance(predictions, list): predictions = np.array(predictions)
    if isinstance(targets, list): targets = np.array(targets)

    # Adjust apnea_threshold_for_epoch to be in units of samples
    apnea_threshold_for_epoch *= sampling_rate

    epoch_level_targets, epoch_level_predictions = [], []
    counter = 0
    # Create signal-level and epoch-level representations
    for epoch in sorted(list(stage_dict.keys())):
        if stage_dict[epoch] in ['R','1','2','3']:
                    
                    if predictions[counter:counter + sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                        epoch_level_predictions.append(1)
                    else:
                        epoch_level_predictions.append(0)
                    
                    if targets[counter:counter + sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                        epoch_level_targets.append(1)
                    else:
                        epoch_level_targets.append(0)
                    counter += sampling_rate*epoch_length
                    if counter >= len(predictions):break

    return np.array(epoch_level_predictions), np.array(epoch_level_targets)