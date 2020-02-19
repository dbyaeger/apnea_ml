#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:26:18 2019

@author: danielyaeger
"""

import numpy as np
import pickle
from pathlib import Path
import pandas as pd
from autoscorer.autoscorer_helpers import convert_to_sequence

def make_rswa_transition_matrix(path_to_p_files: str = '/Volumes/TOSHIBA EXT/artifact_reduced_emg_data', 
                                partitions: list = ['train'], 
                                partition_file_name: str = 'master_ID_list.p',
                                to_save: bool = True, epoch_length: int = 30,
                                sampling_rate: int = 10):
    """Builds transition matrix for P, T, and None events. Only calculates 
    transition matrix for REM epochs in the partitions that DO NOT contain 
    apnea/hypopnea events.
    
    INPUTS:
        path_to_files: path to the directory containing .p files, formatted as
        specified below:
            
            {
                    "ID":ID,
                    "study_start_time":study_start_time,
                    "staging":[(start_time, end_time, stage) for nrem and rem],
                    "apnia_hypopnia_events": [(start_time, end_time, type) w.r.t. apnia_hypopnia_events],
                    "rswa_events":[(start_time, end_time, type) w.r.t. rsw_events],
                    "signals":{"signal_name":[raw_signal] w.r.t. all signals}
                }
            EXPECTS A PARTITION FILE IN THE SAME DIRECTORY AS .P FILES
    
        parititions: list of partitions to use when creating transition matrix,
            by default set to train only.
        
        partition_file_name: name of partition file in path_to_p_files directory
        
    PARAMETERS:
        to_save: set to True to save transition matrix as a .npy file
        
        epoch_length: lenght of epoch in seconds, set to 30 by default
        
        sampling_rate: rate at which RSWA labels are classified, by default set
            to 10
    
    RETURNS:
        transition_df: pandas dataframe with rows that indicate state i and 
        columns that show state i - 1. The count for each transition is shown.
        
        transition_matrix: numpy array. Same information as in transition_df,
            but counts have been converted to transition probabilites.
            
    """
    if not isinstance(path_to_p_files, Path):
        path_to_p_files = Path(path_to_p_files)
        
    with path_to_p_files.joinpath(partition_file_name).open('rb') as fh:
        partition_file = pickle.load(fh)
    
    # Get unique IDs
    unique_IDs = []
    for partition in partitions:
        unique_IDs.extend(list(partition_file[partition]))
    
    # Get files
    files = [f for f in path_to_p_files.iterdir() if f.name.split('_')[0] in unique_IDs]
    
    # Create transition matrix
    label_map = {0:'0', 1:'P', 2:'T'}
    transition_dictionary = {label_map[label]: {label_map[label]: 0 for label in label_map} for label in label_map}
    
    for file in files:
        #print(f'Processing file: {file.name}')
        with file.open('rb') as fh: 
            data = pickle.load(fh)
        
        rem_time = [x for x in data["staging"] if x[-1] == 'R'][0]
        #print(f'\tREM time: {rem_time}')
        rswa_seq = convert_to_sequence(event_list = data["rswa_events"], 
                                       start_time = rem_time[0],
                                       end_time = rem_time[1])
        #print(f'\tLength of rswa_seq: {len(rswa_seq)}')
        
        epoch_start = rem_time[0]//epoch_length
        #print(f'\tepoch_start: {epoch_start}')
        epoch_end = rem_time[1]//epoch_length
        #print(f'\tepoch_end: {epoch_end}')
        
        for i, epoch in enumerate(range(epoch_start,epoch_end)):
            epoch_start_time = epoch*epoch_length
            epoch_end_time = (epoch + 1)*epoch_length
            
            #print(f'\t\tepoch_start_time: {epoch_start_time}')
            #print(f'\t\tepoch_end_time: {epoch_end_time}')
            # Create flag for apnea/hypopnea
            apnea_free = True
            
            # Screen for apnea/hypopnea
            for event in data["apnia_hypopnia_events"]:
                if event[0] <= epoch_end_time <= event[1]: 
                    apnea_free = False
                    #print(f'\t\t\tInterloping A/H event found: {event}')
                elif epoch_start_time <= event[1] <= epoch_end_time:
                    apnea_free = False
                    #print(f'\t\t\tInterloping A/H event found: {event}')
            
            if apnea_free == True:
                for idx in range(i*epoch_length*sampling_rate,(i+1)*epoch_length*sampling_rate):
                    if idx > 0:
                        curr_label = label_map[rswa_seq[idx]]
                        past_label = label_map[rswa_seq[idx-1]]
                        transition_dictionary[curr_label][past_label] += 1

    transition_df = pd.DataFrame.from_dict(transition_dictionary).T
    
    # Convert to normalized numpy array
    transition_matrix = np.array(transition_df)/np.sum(transition_df, axis = 1)[:,np.newaxis]
   
    # Save matrix and normalized initial stage probability
    if to_save:
        np.save(str(path_to_p_files.joinpath(f'rswa_event_transition_matrix')),transition_matrix)
        
    return transition_df, transition_matrix
                        
 
def make_apnea_hypopnea_transition_matrix(path_to_npy_files: str = '/Users/danielyaeger/Documents/raw_apnea_data',
                                                partitions: list = ['train'],
                                                to_pickle: bool = True,
                                                n_classes: int = 2,
                                                sampling_rate: int = 10,
                                                epoch_length: int = 30):
    "Builds transition matrix for apnea/hypoapnea per sample (typically 10 Hz)"
    
    if not isinstance(path_to_npy_files, Path):
        path_to_npy_files = Path(path_to_npy_files)
        
    with path_to_npy_files.joinpath('ID_partitions.p').open('rb') as fh:
        partition_file = pickle.load(fh)
    
    with path_to_npy_files.joinpath('stage_dict.p').open('rb') as fh:
        stage_dict = pickle.load(fh)
        
    # Get unique IDs
    unique_IDs = []
    for partition in partitions:
        unique_IDs.extend(list(partition_file[partition]))
    
    # Get apnea/hypoapnea targets
    with path_to_npy_files.joinpath('apnea_hypopnea_targets.p').open('rb') as fh:
        targets = pickle.load(fh)
    
    assert n_classes in [2,3], "n_classes must be 2 or 3!"
    if n_classes == 3:    
        label_map = {0:'0', 1:'Hypopnea', 2:'Apnea'}
    elif n_classes ==2:
        label_map = {0: '0', 1: 'Hypopnea/Apnea'}
        for ID in unique_IDs:
            idx = np.nonzero(targets[ID] > 1)[0]
            if len(idx) > 0:
                targets[ID][idx] = 1  
    
    transition_dictionary = {label_map[label]: {label_map[label]: 0 for label in label_map} for label in label_map}
    for ID in unique_IDs:
        for epoch in stage_dict[ID]:
            if epoch in ['R','1','2','3']:
                for i,idx in enumerate(np.arange((epoch-1)*sampling_rate*epoch_length:epoch*sampling_rate*epoch_length)):
                    if stage_dict[ID][epoch-1] not in ['R','1','2','3'] and i == 0:
                        continue
                    else:
                        curr_label = label_map[targets[ID][idx]]
                        past_label = label_map[targets[ID][idx-1]]
                        transition_dictionary[curr_label][past_label] += 1

    transition_df = pd.DataFrame.from_dict(transition_dictionary).T
    
    # Convert to normalized numpy array
    transition_matrix = np.array(transition_df)/np.sum(transition_df, axis = 1)[:,np.newaxis]
    print(f'transition matrix:\n {transition_matrix}')
    
    # Save matrix and normalized initial stage probability
    if to_pickle:
        with path_to_npy_files.joinpath(f'apnea_hypopnea_transition_matrix').open('wb') as fout:
            pickle.dump(transition_matrix, fout)
        
    return transition_df, transition_matrix
                

def make_sleep_stage_transition_matrix(path_to_npy_files: str = '/Volumes/Elements/sleep_staging_numpy',
                                                partitions: list = ['train'],
                                                n_classes: int = 5,
                                                to_pickle: bool = True):
    """
    Builds sleep stage transition matrix and initial stage probability distribution
    from selected partitions.
    """
    
    assert n_classes in [2,3,5], f'n_classes must be 2,3 or 5'
    
    if n_classes == 5:
        stage_map = {'W': 'W', '1': '1', '2': '2', '3': '3', 'R': 'R'}
    elif n_classes == 2:
        stage_map = {'W': 'W/N', '1': 'W/N', '2': 'W/N', '3': 'W/N', 'R': 'R'}
    elif n_classes == 3:
        stage_map = {'W': 'W', '1': 'N', '2': 'N', '3': 'N', 'R': 'R'}
    
    inverse_stage_map = {stage_map[stage]:stage for stage in stage_map}
    
    if not isinstance(path_to_npy_files, Path):
        path_to_npy_files = Path(path_to_npy_files)
    
    
    with path_to_npy_files.joinpath('data_partition.p').open('rb') as fh:
        partition_file = pickle.load(fh)
    
    # Get unique IDs
    unique_IDs = []
    for partition in partitions:
        unique_IDs.extend(list(set([f[0].split('_')[0] for f in list(partition_file[partition])])))
        
    # Get stages
    with path_to_npy_files.joinpath('stage_dict.p').open('rb') as fh:
        stage_dict = pickle.load(fh)
        
    transition_dictionary = {sleep_stage: {sleep_stage: 0 for sleep_stage in inverse_stage_map} for sleep_stage in inverse_stage_map}
    initial_stage_prob = {sleep_stage: 0 for sleep_stage in inverse_stage_map}
    
    possible_stages = ['W','1','2','3','R']
    for ID in unique_IDs:
        for idx, epoch in enumerate(sorted(list(stage_dict[ID].keys()))):
            if idx == 0:
                initial_stage_prob[stage_map[stage_dict[ID][epoch]]] += 1
                continue
            else:
                curr_label = stage_dict[ID][epoch]
                past_label = stage_dict[ID][epoch-1]
                if curr_label in possible_stages and past_label in possible_stages:
                    curr_stage = stage_map[curr_label]
                    past_stage = stage_map[past_label]
                    transition_dictionary[curr_stage][past_stage] += 1
                    
    #normalize initial stage dict probability
    for stage in initial_stage_prob: initial_stage_prob[stage]/len(unique_IDs)
        
    
    transition_df = pd.DataFrame.from_dict(transition_dictionary).T
    
    # Convert to normalized numpy array
    transition_matrix = np.array(transition_df)/np.sum(transition_df, axis = 1)[:,np.newaxis]
    
    # Save matrix and normalized initial stage probability
    if to_pickle:
        with path_to_npy_files.joinpath(f'sleep_transition_matrix_{n_classes}_classes').open('wb') as fout:
            pickle.dump(transition_matrix, fout)
        with path_to_npy_files.joinpath(f'initial_sleep_stage_prob_{n_classes}_classes').open('wb') as fout:
            pickle.dump(initial_stage_prob, fout)
    
    return transition_df
                    
            
    
    
    
    
    
        
    
    