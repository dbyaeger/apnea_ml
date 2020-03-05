#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:04:48 2020

@author: danielyaeger
"""
from pathlib import Path
import pickle

def find_intersection_set(path = '/Users/danielyaeger/Documents/raw_a_b'):
    if not isinstance(path, Path): path = Path(path)
    
    directories = [f for f in path.iterdir() if f.is_dir()]
    for i, directory in enumerate(directories):
        IDs = set([f.name.split('.')[0] for f in directory.iterdir() if f.name.startswith('X')])
        if i == 0:
            id_set = IDs
        else:
            id_set = IDs.intersection(id_set)
    return id_set

def trim_files(IDs, path = '/Users/danielyaeger/Documents/raw_a_b'):
    if not isinstance(path, Path): path = Path(path)
    
    directories = [f for f in path.iterdir() if f.is_dir()]
    
    for directory in directories:
        for file in directory.iterdir():
            if file.name.startswith('X'):
                if file.name.split('.')[0] not in IDs:
                    file.unlink()

def trim_targets(path = '/Users/danielyaeger/Documents/raw_a_b'):
    if not isinstance(path, Path): path = Path(path)
    
    directories = [f for f in path.iterdir() if f.is_dir()]
    
    for directory in directories:
        IDs_in_dir = [f.name.split('.')[0] for f in directory.iterdir() if f.name.startswith('X')]
        
        with directory.joinpath('apnea_hypopnea_targets.p').open('rb') as fh:
            targets = pickle.load(fh)
        
        new_targets = {}
        
        for ID in targets:
            if ID in IDs_in_dir:
                new_targets[ID] = targets[ID]
        
        with directory.joinpath('apnea_hypopnea_targets.p').open('wb') as fh:
            pickle.dump(new_targets,fh)

def detect_apnea(path = '/Users/danielyaeger/Documents/datasets_a_b/baselined_features_no_hr_0.67'):
    if not isinstance(path, Path): path = Path(path)
    
    with path.joinpath('apnea_hypopnea_targets.p').open('rb') as fh:
        targets = pickle.load(fh)
        
    for ID in targets:
        print(f'ID: {ID}\t Number of apneic/hypopneic samples: {len(np.nonzero(targets[ID] > 0)[0])}')