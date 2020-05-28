#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:21:46 2020

@author: danielyaeger
"""
from pathlib import Path
from test.evaluate import evaluate
from test.predict_from_DNN import predict
from test.maxlikelihood import maxlikelihood
from viterbi.viterbi_wrapper import smooth_all_with_viterbi
from get_epoch_level_predictions import make_apnea_dict, make_ground_truth_apnea_dict

def evaluate_models_and_generate_predictions(
        prediction_configurations: list,
        path_to_ground_truth_staging: str,
        ground_truth_staging_name: str,
        data_path_for_ground_truth: str,
        save_path: str,
        save_evaluation_name: str,
        prediction_method_for_pipeline: str = 'viterbi'):
    """
    Creates ground truth apnea labels at epoch level if they do not exist.
    Then gets predictions for each prediction set in prediction_configuration
    and 
        1) gets max-likelihood and viterbi-smoothed predictions
        2) creates full-length epoch-level observations using 
            prediction_method_for_pipeline (viterbi or max likelihood)
            to be used for pipeline
        3) evaluates each prediction file at signal and epoch levels
    """
    assert prediction_method_for_pipeline in['max_likelihood', 'viterbi'], \
        'prediction_method_for_pipeline must be max_likelihood or viterbi!'
    
    # Create full-length ground-truth apnea labels for pipeline if they don't exist
    if not save_path.joinpath('ground_truth_apnea_dict.p').exists():
        make_ground_truth_apnea_dict(path_to_data = data_path_for_ground_truth,
                                     path_to_ground_truth_staging = path_to_ground_truth_staging,
                                     ground_truth_staging_name = ground_truth_staging_name,
                                     save_path = save_path)
    
    # Generate predictions and create data dictionary to input to evaluate method
    
    data_dictionary = []
    
    for prediction in prediction_configurations:
        if not Path(prediction['save_path']).joinpath(prediction['save_name_signal_level']).exists():
            
            predict(path_to_model = prediction['path_to_model'],
                    path_to_results = prediction['path_to_results'],
                    path_to_data = prediction['path_to_data'],
                    model_name = prediction['model_name'],
                    save_path = prediction['save_path'],
                    save_name = prediction['save_name_signal_level'],
                    verbose = True)
        
        if not Path(prediction['save_path']).joinpath(prediction['save_name_signal_level'].split('.') + '_ML.p').exists():
            
            maxlikelihood(path_to_probabilities = prediction['save_path'],
                      save_path = prediction['save_path'],
                      probablities_file_name = prediction['save_name_signal_level'],
                      save_name = prediction['save_name_signal_level'].split('.') + '_ML.p')
            
        data_dictionary.append({'data_set_name': prediction['save_name_signal_level'].split('.') + '_ML.p',
                            'data_set_path': prediction['save_path'],
                            'data_set_identifer': prediction['save_name_signal_level'].split('.') + '_ML.p',
                            'data_set_metadata_path': prediction['path_to_data']})
        
        if not Path(prediction['save_path']).joinpath(prediction['save_name_signal_level'].split('.') + '_HMM.p').exists():
            
            smooth_all_with_viterbi(path_to_probabilites = prediction['save_path'],
                            save_path = prediction['save_path'],
                            data_path = prediction['path_to_data'],
                            probabilities_file_name = prediction['save_name_signal_level'],
                            save_name = prediction['save_name_signal_level'].split('.') + '_HMM.p',
                            stage_file = 'stage_dict.p',
                            REM_only = False)
        
        data_dictionary.append({'data_set_name': prediction['save_name_signal_level'].split('.') + '_HMM.p',
                            'data_set_path': prediction['save_path'],
                            'data_set_identifer': prediction['save_name_signal_level'].split('.') + '_HMM.p',
                            'data_set_metadata_path': prediction['path_to_data']})
        
        # Make epoch-level predictions
        if prediction_method_for_pipeline == 'max_likelihood':
            signal_level_predictions_name  = prediction['save_name_signal_level'].split('.') + '_ML.p'
        else:
            signal_level_predictions_name  = prediction['save_name_signal_level'].split('.') + '_HMM.p'
        
            make_apnea_dict(signal_level_predictions_name = signal_level_predictions_name,
                            predictions_path = prediction['save_path'],
                            stage_file_name = 'stage_dict.p',
                            stage_path = prediction['path_to_data'],
                            save_name = prediction['save_name_epoch_level'],
                            save_path = save_path)

    # Evaluate
    evaluate(data_dictionary = data_dictionary,
             save_path = save_path,
             save_name = save_evaluation_name,
             name_of_ground_truth_staging = ground_truth_staging_name,
             path_to_ground_truth_staging = path_to_ground_truth_staging,
             name_of_ground_truth_apneas = 'ground_truth_apnea_dict.p',
             path_to_ground_truth_apneas = save_path)
    
    
    
    