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
from data_generators.data_generator_apnea_ID_batch import DataGeneratorApneaAllWindows
from utils.transition_matrix_tools import make_apnea_hypopnea_transition_matrix

def evaluate_and_predict(path_to_model: str, path_to_results: str, 
                         path_to_data: str, model_name: str = 'five_conv_two_dense'):
    """Loads the trials object and finds the model with the lowest loss. Loads
    the corresponding trained model and evaluates it over every model in the 
    test set, saving a dictionary of dictionaries keyed by ID with predictions, 
    true labels, confusion matrix, and balanced accuracy."""
    
    
    if not isinstance(path_to_results, Path):
        path_to_results = Path(path_to_results)
    
    with path_to_results.joinpath(model_name + '_trials').open('rb') as fh:
        results = pickle.load(fh)
    
    # Get the index of the saved best model
    results = sorted(results, key = lambda x: x['loss'])
    best_model_idx = results[0]['iteration']
    
    # Load the best model
    if not isinstance(path_to_model, Path):
        path_to_model = Path(path_to_model)
    
    model_path = str(path_to_model.joinpath(f'{model_name}_{best_model_idx}hdf5'))
    print(f'Best model path: {model_path}')
    
    best_model = load_model(model_path)
    
    # Set up data generator
    test_gen = DataGeneratorApneaAllWindows(n_classes = 2,
                                    path_to_data = path_to_data,
                                    mode="test",
                                    context_samples=300,
                                    shuffle = False,
                                    use_staging = True)
    IDs = test_gen.IDs
    test_results_dict = {ID: {} for ID in IDs}
    
    # iterate over IDs and generate predictions
    for ID in IDs:
        X, y = test_gen.__data_generation(ID)
        y_pred = best_model.predict(X)
        test_results_dict[ID]['targets'] = y
        test_results_dict[ID]['predictions'] = y
        
        try:
            test_results_dict[ID]['balanaced_accuracy'] = balanced_accuracy_score(y.argmax(-1),
                                                      y_pred.argmax(-1))
        except:
             test_results_dict[ID]['balanaced_accuracy'] = np.nan
        
        test_results_dict[ID]["confusion_matrix"] = confusion_matrix(y.argmax(-1),y_pred.argmax(-1))
        
    with path_to_results.joinpath(f'{model_name}_test_set_results.p').open('wb') as fh:
        pickle.dump(test_results_dict, fh)

#def correct_priors(results_path: str, results_file_name: str, prior_correction_method: callable):
#    """Corrects the priors and saves 
#    """
#
#def smooth_posteriors(results_path: str, results_file_name: str, path_to_data: str):
#    """Smoothes posteriors using viterbi search. Also creates transition probability
#    matrix if one does not exist in path_to_data. Saves the smoothed"""


        
        
        
        
        
        
    
    
    
    