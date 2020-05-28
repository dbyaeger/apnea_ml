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
from sklearn.metrics import (balanced_accuracy_score, accuracy_score)
import re
from get_epoch_length_predictions import get_epoch_level_predictions

def evaluate(data_dictionary: list, save_path: str, save_name: str,
             name_of_ground_truth_staging: str,
             path_to_ground_truth_staging: str,
             name_of_ground_truth_apneas: str,
             path_to_ground_truth_apneas: str,
             metrics: list = [balanced_accuracy_score, accuracy_score],
             apnea_threshold_for_epoch: float = 10, sampling_rate: int = 10, 
             epoch_length: int = 30) -> pd.DataFrame:
    """Evaluates the data in the data dictionary using the supplied list of metrics.
    Each entry in the data dictionary should have the following format:
        
        data_dictionary = [{'data_set_name': <string>,
                            'data_set_path': <string>,
                            'data_set_identifer': <string>,
                            'data_set_metadata_path': <string>,}]
    
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
    
    NOTE: signal-level accuracies ONLY make sense when ground-truth staging
    is used.
    
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
    
    if not isinstance(path_to_ground_truth_apneas, Path):
        path_to_ground_truth_apneas = Path(path_to_ground_truth_apneas)
        
    # Make list of metric names
    metric_names = []
    for metric in metrics:
        metric_name = re.findall(r'function (.*) at', str(metric))[0]
        for evaluation_type in ['signal','epoch']:
            metric_names.append(f'{metric_name}_{evaluation_type}')
            
    
    # Dictionary to hold results
    results = {metric_name: [] for metric_name in metric_names}
    results['data'] = []
    
    # Get ground truth apneas
    with path_to_ground_truth_apneas.joinpath(name_of_ground_truth_apneas).open('rb') as gt:
        true_apnea_dict = pickle.load(gt)
    
    # Get predictions and targets
    for data_set in data_dictionary:
        results['data'].append(data_set['data_set_identifer'])
        #print(f'data_set: {data_set}')
        # Make dictionary to hold results for each ID
        results_dict_IDs = {metric_name: [] for metric_name in metric_names}
        
        # Open data file containing predictions
        with Path(data_set['data_set_path']).joinpath(data_set['data_set_name']).open('rb') as fin:
            data = pickle.load(fin)
        for ID in data:
            epoch_preds_dict = get_epoch_level_predictions(ID = ID,
                                                  predictions = data[ID]['predictions'],
                                                  data_path = data_set['data_set_metadata_path'],
                                                  stage_file = 'stage_dict.p',
                                                  apnea_threshold_for_epoch = apnea_threshold_for_epoch,
                                                  sampling_rate = sampling_rate,
                                                  epoch_length = epoch_length) 
            
            # Convert epoch_level predictions and targets into lists
            epoch_preds_list = [epoch_preds_dict[epoch] for epoch in sorted(list(epoch_preds_dict.keys()))]
            epoch_targets_list = [true_apnea_dict[epoch] for epoch in sorted(list(true_apnea_dict.keys()))]
            assert len(epoch_preds_list) == len(epoch_targets_list), \
            f'epoch_preds_list length ({len(epoch_preds_list)}) != epoch_targets_list length ({len(epoch_targets_list)})!'
            
            for metric in metrics:
                metric_name = re.findall(r'function (.*) at', str(metric))[0]
                results_dict_IDs[f'{metric_name}_signal'].append(metric(data[ID]['targets'],
                                                                        data[ID]['predictions']))
                results_dict_IDs[f'{metric_name}_epoch'].append(metric(epoch_targets_list,
                                                                        epoch_preds_list))
        # Now collapse into mean and std dev.
        for metric_name in metric_names:
            mean = round(np.mean(results_dict_IDs[metric_name]),3)
            std = round(np.std(results_dict_IDs[metric_name]),3)
            results[metric_name].append(f'{mean} ± {std}')
        
    # Make into a dataframe and save as a .csv file
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(save_path.joinpath(save_name), index = False)
    
    return results_df
    