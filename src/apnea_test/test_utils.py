#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:19:23 2020

@author: danielyaeger
"""
from pathlib import Path
import pickle
#from apnea_test import Main
from main_with_online_baselining import Main
import numpy as np
import gc
def test_wrapper(path = '/Users/danielyaeger/Documents/raw_a_b', K_FOLD = 3,
                 out_path = '/Users/danielyaeger/Documents/raw_a_b'):
    if not isinstance(path, Path): path = Path(path)
    if not isinstance(out_path, Path): out_path = Path(out_path)
    directories = [f for f in path.iterdir() if f.is_dir()]
    
    results_dict = {}
    for directory in directories:
        print(f'Analyzing {directory.name}')
        IDs = set([f.name.split('.')[0] for f in directory.iterdir() if f.name.startswith('X')])
        bal_acc = np.zeros(len(IDs))
        for i, ID in enumerate(IDs):
            tester = Main(ID=ID, K_FOLD=K_FOLD, data_path = str(directory))
            bal_acc_for_ID = tester.main()
            bal_acc[i] = bal_acc_for_ID
            print(f'MEAN BALANCED ACCURACY FOR {ID}: {bal_acc_for_ID}')
            del tester
            gc.collect()          
        results_dict[directory.name] = np.mean(bal_acc)
        with out_path.joinpath('results_dict').open('wb') as fh:
            pickle.dump(results_dict,fh)
    
    
    return results_dict

def test_wrapper_one_dir(path = '/Users/danielyaeger/Documents/raw_no_baseline', K_FOLD = 4,
                 out_path = '/Users/danielyaeger/Documents/raw_no_baseline'):
    if not isinstance(path, Path): path = Path(path)
    if not isinstance(out_path, Path): out_path = Path(out_path)
    IDs = [f.name.split('.')[0] for f in path.iterdir() if f.suffix == '.npy']
    bal_acc = np.zeros(len(IDs))
    for i, ID in enumerate(IDs):
        tester = Main(ID=ID, K_FOLD=K_FOLD, data_path = str(path))
        bal_acc_for_ID = tester.main()
        bal_acc[i] = bal_acc_for_ID
        print(f'MEAN BALANCED ACCURACY FOR {ID}: {bal_acc_for_ID}')
        del tester
        gc.collect()
    print(f'OVERALL MEAN BALANCED ACCURACY: {np.mean(bal_acc)}')         
    with out_path.joinpath('results').open('wb') as fh:
        pickle.dump(bal_acc,fh)
    return bal_acc

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
    
            
if __name__ == "__main__":
#    LOCAL = True
#    
#    if LOCAL:
#        path = '/Users/danielyaeger/Documents/raw_a_b'
#        out_path = '/Users/danielyaeger/Documents/raw_a_b'
#    else:
#        path = '/floyd/input/data'
#        out_path = '/floyd/home'
        
    results = test_wrapper_one_dir()
    
        
    
    