#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:17:26 2020

@author: danielyaeger
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.metrics
import re

def evaluate(data_dictionary: list, metrics: list, data_path: str,
             stage_file: str, save_path: str, save_name: str,
             apnea_threshold_for_epoch: float = 0.1, sampling_rate: int = 10, 
             epoch_length: int = 30) -> pd.DataFrame:
    """Evaluates the data in the data dictionary using the supplied list of metrics.
    Each entry in the data dictionary should have the following format:
        
        data_dictionary = [{'data_set_name': <string>,
                            'data_set_path': <string>,
                            'data_set_identifer': <string>}]
    
    where data_set_name is the name of the data set file in the data_set_path,
    and data_set_identifier is the name that should be given to the dataset
    in the output pandas table.
    
    Each data_set file is assumed to be keyed by ID, with subdictionaries key
    as 'predictions' and 'targets' with the corresponding predictions and targets.
    
    For each data_set, metrics at both the signal-level and epoch-level will be
    calculated. For the epoch-level, only non-Wake epochs will be considered.
    An epoch will be considered to be an apnea epoch if the total length of the
    apneic events exceeds the apnea_threshold_for_epoch (which is in units of 
    seconds).
    
    INPUTS:
        data_dictionary: see above for descrption.
        metrics: list of metrics to apply on sample and epoch level.
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
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
        
    # Make list of metric names
    metric_names = []
    for metric in metrics:
        metric_name = re.fndall(re.findall(r'function (.*) at', str(metric))[0])
        for evaluation_type in ['signal','epoch']:
            metric_names.append(f'{metric_name}_{evaluation_type}')
            
    
    # Dictionary to hold results
    results = {metric_name: [] for metric_name in metric_names}
    results['data'] = []
    
    # Get predictions and targets
    for data_set in data_dictionary:
        results['data'].append(data_set['data_set_identifer'])
        
        # Make dictionary to hold results for each ID
        results_dict_IDs = {metric_name: [] for metric_name in metric_names}
        
        # Open data file containing predictions
        with Path(data_set['data_set_path']).joinpath(data_set['data_set_name']).open('rb') as fin:
            data = pickle.load(fin)
        for ID in data:
            ID_dict = get_targets_and_predictions(ID = ID,
                                                  predictions = data[ID]['predictions'],
                                                  data_path = data_path,
                                                  stage_file = stage_file,
                                                  apnea_threshold_for_epoch = apnea_threshold_for_epoch,
                                                  sampling_rate = sampling_rate,
                                                  epoch_length = epoch_length)
            
            for metric in metrics:
                metric_name = re.fndall(re.findall(r'function (.*) at', str(metric))[0])
                results_dict_IDs[f'{metric_name}_signal'].append(metric(ID_dict['signal_level_targets'],
                                                                        ID_dict['signal_level_predictions']))
                results_dict_IDs[f'{metric_name}_epoch'].append(metric(ID_dict['epoch_level_predictions'],
                                                                        ID_dict['epoch_level_targets']))
        # Now collapse into mean and std dev.
        for metric_name in metric_names:
            results.append(f'{np.mean(results_dict_IDs[metric_name])} ± {np.std(results_dict_IDs[metric_name])}')
        
    # Make into a dataframe and save as a .csv file
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(save_path.joinpath(save_name), index = False)
    
    return results_df
                
                
            
                
                
            
            
        
        

def get_targets_and_predictions(ID: str,
                                predictions: np.ndarray,
                                data_path: str,
                                stage_file: str,
                                apnea_threshold_for_epoch: float, 
                                sampling_rate: int = 10, 
                                epoch_length: int = 30) -> dict:
    """Takes in ID and arrays of predictions and targets and returns
    signal-level predictions and targets and epoch-level predictions and targets
    NPUTS:
        ID: sleeper ID.
        predictions: array of full-length predictions
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
    
    with data_path.joinpath(stage_file) as fs:
        stage_dict = pickle.load(fs)[ID]
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
        targets = pickle.load(ta)[ID]
        
    if REM_only:
        stages = ['R']
    else:
        stages = ['R','1','2','3']
    
    signal_level_targets, signal_level_predictions = [], []
    epoch_level_targets, epoch_level_predictions = [], []
    
    # Create signal-level and epoch-level representations
    for epoch in stage_dict:
        if stage_dict[epoch] in stages:
                    signal_level_predictions.append(predictions[(epoch-1)*sampling_rate*epoch_length:epoch*sampling_rate*epoch_length])
                    signal_level_targets.append(targets[(epoch-1)*sampling_rate*epoch_length:epoch*sampling_rate*epoch_length])
                    
                    if predictions[(epoch-1)*sampling_rate*epoch_length:epoch*sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                        epoch_level_predictions.append(1)
                    else:
                        epoch_level_predictions.append(0)
                    
                    if targets[(epoch-1)*sampling_rate*epoch_length:epoch*sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                        epoch_level_targets.append(1)
                    else:
                        epoch_level_targets.append(0)
    
    return {'signal_level_predictions': np.array(signal_level_predictions),
            'signal_level_targets': np.array(signal_level_targets),
            'epoch_level_predictions': np.array(epoch_level_predictions),
            'epoch_level_targets': np.array(epoch_level_targets)}
        
    