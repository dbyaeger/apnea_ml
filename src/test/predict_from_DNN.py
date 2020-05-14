#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:41:57 2020

@author: danielyaeger
"""
import pickle
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from data_generators.data_generator_apnea import DataGeneratorApnea

def predict(path_to_model: str, path_to_results: str, 
            path_to_data: str, model_name: str = 'five_conv_two_dense',
            verbose: bool = True):
    """Loads the trials object and finds the model with the lowest loss. Loads
    the corresponding trained model and evaluates it over every model in the 
    test set, saving a dictionary of dictionaries keyed by ID with predictions, 
    true labels, confusion matrix, and balanced accuracy."""
    
    
    if not isinstance(path_to_results, Path):
        path_to_results = Path(path_to_results)
    
    with path_to_results.joinpath(model_name + '_trials').open('rb') as fh:
        results = pickle.load(fh)
    results = results.results

    # Get the index of the saved best model
    results = sorted(results, key = lambda x: x['loss'])
    best_model_idx = results[0]['iteration']
    
    # Load the best model
    if not isinstance(path_to_model, Path):
        path_to_model = Path(path_to_model)
    
    model_path = str(path_to_model.joinpath(f'{model_name}_{best_model_idx}.hdf5'))
    if verbose:
        print(f'Best model path: {model_path}')
    
    best_model = load_model(model_path)
    
    # Set up data generator
    test_gen = DataGeneratorApnea(data_path = path_to_data, mode="test")         
    IDs = test_gen.IDs
    if verbose:
        print(f'Number of IDs in test set: {len(IDs)}')        
    test_results_dict = {ID: {} for ID in IDs}
    
    # iterate over IDs and generate predictions
    for ID in IDs:
        if verbose:
            print(f'ID:\t{ID}')
        test_gen = DataGeneratorApnea(n_classes = 2,
                                 data_path = path_to_data,
                                 single_ID = ID,
                                 batch_size = 16,
                                 mode="test",
                                 context_samples=300,
                                 shuffle = False,
                                 use_staging = True,
                                 REM_only = False)
        y = test_gen.labels
        y_pred = best_model.predict_generator(generator = test_gen,
                                              workers=4, 
                                              use_multiprocessing=True, 
                                              verbose=1)
        y = y[:len(y_pred)]
        test_results_dict[ID]['targets'] = y
        test_results_dict[ID]['predictions'] = y_pred
        
        try:
            test_results_dict[ID]['balanaced_accuracy'] = balanced_accuracy_score(y,
                                                      y_pred.argmax(-1))
        except:
             test_results_dict[ID]['balanaced_accuracy'] = np.nan
        
        test_results_dict[ID]["confusion_matrix"] = confusion_matrix(y,y_pred.argmax(-1))
        
        if verbose:
            print(f"Balanced accuracy: {test_results_dict[ID]['balanaced_accuracy']}")
    
    with path_to_results.joinpath(f'{model_name}_test_set_results.p').open('wb') as fh:
        pickle.dump(test_results_dict, fh)

        
        
        
        
        
    
    
    
    