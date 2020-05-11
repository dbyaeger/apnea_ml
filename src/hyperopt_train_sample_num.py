#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:03:37 2020

@author: danielyaeger
"""

from pathlib import Path
import numpy as np
import csv
import pickle
import ast
from model_functions.model_functions import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score
from data_generators.data_generator_apnea import DataGeneratorApnea
from hyperopt.hp import uniform, loguniform, quniform, lognormal, choice
from hyperopt import STATUS_OK
from hyperopt import Trials, tpe, fmin

class HyperOptimizer():
    """
    Wrapper class for use with hyperopt package. Must be rewritten for each
    model.
    
    INPUTS:
        data_path: where data files live
        model_path: where to save model
        results_path: where to save results
        model_specs_path: where the results of bayesian optimization for the model to be used live
        model_name: name of the model to be used
        experiment_name: name of files
        metric: metric to be optimizied on validation set
        max_evals: how many iterations are allowed for finding best parameters
        variables: list of variables to optimize
        distributions: distributions to use for each variable (by order)
        arguments: arguments to each distribution-generating function as tuple
        
    Builds a .csv file at the path specified by results_path which can be used to
    find the best model parameters.
    
    Also has a results attribute which can also be used to find the best model
    parameters.
    """
    def __init__(self, data_path: str, model_path: str, results_path: str,
                 model_specs_path: str, model_name: str,
                 experiment_name: str = 'five_conv_two_dense_best_num',
                 metric: callable = balanced_accuracy_score,
                 max_evals: int = 100,
                 variables: list = ['desired_number_of_samples'],
                 distributions: list = ['quniform'],
                 arguments: list = [(0.5e6,30e6,1)]):

        self.data_path = self.convert_to_path(data_path)
        self.model_path = self.convert_to_path(model_path)
        self.results_path = self.convert_to_path(results_path)
        self.model_specs_path = self.convert_to_path(model_specs_path).joinpath(model_name + '_trials')
        self.save_name = experiment_name
        self.trial_path = self.results_path.joinpath(self.save_name + '_trials')
        self.csv_path = self.results_path.joinpath(experiment_name + '.csv')
        
        if not self.csv_path.is_file():
            with self.csv_path.open('w') as fh:
                writer = csv.writer(fh)
                writer.writerow(['loss', 'params', 'iteration'])
            
        self.max_evals = max_evals
        self.metric = metric
        
        if self.trial_path.is_file():
                print('Retrieving trials object')
                with self.trial_path.open('rb') as fh:
                    self.bayes_trials = pickle.load(fh)
                self.iteration = len(self.bayes_trials)
                
        else:
            self.bayes_trials = Trials()
            self.iteration = 0
        
        # How many steps to run before printing/saving
        self.step = 1
        
        # create domain space
        self.space = self.create_domain_space(variables,distributions,arguments)
    
    def objective(self, params):
        """Objective function for Hyperparameter optimization
        """
        self.iteration += 1

        # make sure parameters that must be integers are integers
        params['desired_number_of_samples'] = int(params['desired_number_of_samples'])
        
        # make metric_result negative for optimization
        metric_result = self.run_with(**params)
        loss = 1 - metric_result
        
        # write results to csv file
        with self.csv_path.open('a') as fh:
            writer = csv.writer(fh)
            writer.writerow([loss, params, self.iteration])
        
        return {'loss': loss, 'params': params, 'iteration': self.iteration,
                'status': STATUS_OK}
    
    def optimize_params(self):
        """Wrapper method for fmin function in hyperopt package. Uses previously
        stored results if available
        """
        i = 1
        while i < (self.max_evals + 1):
            if self.trial_path.is_file():
                print('Retrieving trials object')
                with self.trial_path.open('rb') as fh:
                    self.bayes_trials = pickle.load(fh)
                i = len(self.bayes_trials) + 1
                self.optimize_helper(i)
            else:
                print('No trials object found')
                self.optimize_helper(i)
    
    def optimize_helper(self, max_evals: int):
        """ Wrapper method for fmin function in hyperopt package 
        """
        best = fmin(fn = self.objective, 
                    space = self.space, 
                    algo = tpe.suggest, 
                    trials = self.bayes_trials, 
                    max_evals = max_evals,
                    verbose = 0)
        print(best)
        
        with self.trial_path.open('wb') as fh:
            pickle.dump(self.bayes_trials, fh)
    
    def run_with(self,desired_number_of_samples):
        """Builds a 1-conv layer, 2-dense layer neural net with specified parameters
        and trains. Returns metric result on cross val set.
        """
       
        #Make generators
        train_generator = DataGeneratorApnea(n_classes = 2,
                                    data_path = self.data_path,
                                    batch_size = 128,
                                    mode="train",
                                    context_samples=300,
                                    shuffle = True,
                                    desired_number_of_samples = desired_number_of_samples,
                                    load_all_data = True)
        
        cv_generator =  DataGeneratorApnea(
                                    n_classes = 2,
                                    data_path = self.data_path,
                                    batch_size = 128,
                                    mode="cv",
                                    context_samples=300,
                                    shuffle = True,
                                    load_all_data = True)
        # Print class weights
        print(f'Class weights for training: {train_generator.class_weights}')
        print(f'Training data length: {len(train_generator})')
        
        
        # Set parameters
        model_path = str(self.model_path.joinpath(f'{self.save_name}_{self.iteration}.hdf5'))                            
        learning_rate = 1e-3
        n_epoch = 30
        stopping = EarlyStopping(patience=5)

        reduce_lr = ReduceLROnPlateau(factor=0.1,
                                        patience=8,
                                        min_lr=1e-6)
        
        model_checkpoint = ModelCheckpoint(filepath=model_path, 
                                             monitor='loss', 
                                             save_best_only=True)
        
        # Get best parameters
        with self.model_specs_path.open('rb') as fh:
            results = pickle.load(fh)
        results = results.results

        # Get the index of the saved best model
        results = sorted(results, key = lambda x: x['loss'])
        best_model_params = ast.literal_eval(results[0]['params'])
        
        #Unpack parameters
        model = self.build_model_from_params(best_model_params,
                                             train_generator.dim,
                                             learning_rate,
                                             train_generator.n_classes)
        model.summary()
        
        # Train
        model.fit_generator(generator=train_generator,
                  validation_data=cv_generator,
                  use_multiprocessing=True,
                  workers=4,
                  epochs=n_epoch,
                  class_weight=train_generator.class_weights,
                  callbacks=[stopping, reduce_lr, model_checkpoint],
                  verbose=2)
        
        # Evaluate balanced accuracy on best model
        best_model = load_model(model_path)
        cv_generator =  DataGeneratorApnea(n_classes = 2,
                                    data_path = self.data_path,
                                    batch_size = 128,
                                    mode="cv",
                                    context_samples=300,
                                    shuffle=False,
                                    load_all_data = True)
        y_pred = best_model.predict_generator(cv_generator)
        y_true = cv_generator.labels[:len(y_pred)]
        
        score = self.metric(y_true, y_pred.argmax(-1))
        
        return score
    
    @property
    def results(self):
        return self.bayes_trials.results
    
    def build_model_from_params(best_model_params, dimension, learning_rate,
                                n_classes):
        """ Build a model and returns a model using the specified parameter
        dictionary"""
    
    params = {"input_shape": dim, "lstm_layers": [], 
              "learning_rate":learning_rate, "conv_layers": [],
              "fc_layers": []}
    try:
        params["fc_layers"].extend([(best_model_params['fc_neurons'],
                                    best_model_params['fc_layer_lambda'],
                                    0.5),
                                    (n_classes,None,None)])
    except:
        params["fc_layers"].append((n_classes,None,None))
    
    # build conv layers
    for layer_num in ['one','two','three','four','five','six','seven']:
        try:
            if layer_num == 'one':
                params["conv_layers"].append((params['conv_filter_num_one'],
                                              params['conv_filter_size_one'],
                                              params['conv_layer_lambda_one'],
                                              True))
            else:
                params["conv_layers"].append((params[f'conv_filter_num_{layer_num}'],
                                              params[f'conv_filter_size_{layer_num}'],
                                              params[f'conv_layer_lambda_{layer_num}'],
                                              False))
        except:
            continue
    
    model = build_model(**params)
    return model
        
    
    @staticmethod
    def create_domain_space(variables: list = ['C', 'gamma'], 
                     distributions: list = ['loguniform','loguniform'], 
                     arguments: list = [(0.1, 10),(1e-6,10)]):
        """ Returns dictionary keyed by variable with the distribution
            
            INPUTS:
                variables: list of string variables
                
                distributions: list of string names of functions to generate 
                distributions
                
                arguments: list of tuples where each tuple contains arguments
                to distribution-generating function in order
            
            RETURNS:
                dictionary keyed by parameter type with specified distribution
                functions as keys
        """
        space = {}
        for i,variable in enumerate(variables):
            if distributions[i] == 'loguniform':
                (low, hi) = arguments[i]
                low, hi = np.log(low), np.log(hi)
                space[variable] = loguniform(variable,low,hi)
            elif distributions[i] == 'lognormal':
                (mu, sigma) = arguments[i]
                space[variable] = lognormal(variable,mu,sigma)
            elif distributions[i] == 'uniform':
                (low, high) = arguments[i]
                space[variable] = uniform(variable,low,high)
            elif distributions[i] == 'quniform':
                (low, high, q) = arguments[i]
                space[variable] = quniform(variable,low,high,q)
        return space
    
    @staticmethod
    def convert_to_path(path: str, make_directory: bool = True):
        """Converts an input string to path and creates the directory if it
        does not already exist"""
        if not isinstance(path, Path): path = Path(path)
        
        if make_directory:
            if not path.is_dir():
                path.mkdir()
        
        return path


        