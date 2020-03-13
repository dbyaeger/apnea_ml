#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:20:03 2020

@author: danielyaeger
"""
from pathlib import Path
import numpy as np
import csv
import pickle
#import talos as ta
from model_functions.model_functions import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score
from data_generators.data_generator_apnea_ID_batch import DataGeneratorApneaIDBatch
from data_generators.data_generator_apnea import DataGeneratorApnea
from hyperopt.hp import uniform, loguniform, quniform, lognormal
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
        experiment_name: name of files
        metric: metric to be optimizied on validation set
        savepath: path or string to where results should be saved
        max_evals: how many iterations are allowed for finding best parameters
        variables: list of variables to optimize
        distributions: distributions to use for each variable (by order)
        arguments: arguments to each distribution-generating function as tuple
        
    Builds a .csv file at the path specified by savepath which can be used to
    find the best model parameters.
    
    Also has a results attribute which can also be used to find the best model
    parameters.
    """
    def __init__(self, data_path: str, model_path: str, results_path: str,
                 experiment_name: str = 'one_conv_two_dense',
                 metric: callable = balanced_accuracy_score,
                 max_evals: int = 50,
                 variables: list = ['dropout_rate', 'conv_layer_lambda', 'conv_filter_num',
                                    'conv_filter_size', 'fc_neurons', 'fc_layer_lambda'],
                 distributions: list = ['uniform','loguniform','quniform',
                                        'quniform','quniform', 'loguniform'],
                 arguments: list = [(0, 1),(1e-12,1),(16,256,1), (10,300,1),
                                    (32,1024,1), (1e-12,1)]):

        self.data_path = self.convert_to_path(data_path)
        self.model_path = self.convert_to_path(model_path)
        self.results_path = self.convert_to_path(results_path)
        self.save_name = experiment_name
        self.trial_path = self.results_path.joinpath(self.save_name + '_trials')
        self.csv_path = self.results_path.joinpath(experiment_name + '.csv')
        
        if not self.csv_path.isfile():
            with self.savepath.open('w') as fh:
                writer = csv.writer(fh)
                writer.writerow(['loss', 'params', 'iteration'])
            
        self.max_evals = max_evals
        self.metric = metric
        self.iteration = 0
        self.bayes_trials = Trials()
        
        # How many steps to run before printing/saving
        self.step = 1
        
        # create domain space
        self.space = self.create_domain_space(variables,distributions,arguments)
        
    def objective(self, params):
        """Objective function for Hyperparameter optimization
        """
        self.iteration += 1
        
        # make sure parameters that must be integers are integers
        for parameter_name in ['conv_filter_num', 'conv_filter_size', 'fc_neurons']:
            params[parameter_name] = int(params[parameter_name])
        
        
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
            if self.trial_path.isfile():
                with self.trial_path.open('rb') as fh:
                    self.bayes_trials = pickle.load(fh)
                i = len(self.bayes_trials) + 1
                self.optimize_helper(i)
            else:
                self.optimize_helper(i)
    
    def optimize_helper(self, max_evals: int):
        """ Wrapper method for fmin function in hyperopt package 
        """
        best = fmin(fn = self.objective, 
                    space = self.space, 
                    algo = tpe.suggest, 
                    trials = self.bayes_trials, 
                    max_evals = max_evals)
        print(best)
        
        with self.trial_path.open('wb') as fh:
            pickle.dump(self.bayes_trials, fh)
                
    
    def run_with(self,dropout_rate, conv_layer_lambda, conv_filter_num, 
                 conv_filter_size, fc_neurons, fc_layer_lambda):
        """Builds a 1-conv layer, 2-dense layer neural net with specified parameters
        and trains. Returns metric result on cross val set.
        """
       
        #Make generators
        train_generator = DataGeneratorApneaIDBatch(n_classes = 2,
                                    data_path = self.data_path,
                                    batch_size = 128,
                                    mode="train",
                                    context_samples=300,
                                    shuffle = True,
                                    desired_number_of_samples = 2.1e6)
        
        cv_generator =  DataGeneratorApnea(n_classes = 2,
                                    data_path = self.data_path,
                                    batch_size = 128,
                                    mode="cv",
                                    context_samples=300,
                                    load_all_data = True,
                                    shuffle=True)
        
        # Set parameters
        model_path = str(self.model_path.joinpath(f'{self.save_name}_{self.iterations}.hdf5'))                            
        learning_rate = 1e-3
        n_epoch =20
        stopping = EarlyStopping(patience=5)

        reduce_lr = ReduceLROnPlateau(factor=0.1,
                                        patience=8,
                                        min_lr=1e-6)
        
        model_checkpoint = ModelCheckpoint(filepath=model_path, 
                                             monitor='loss', 
                                             save_best_only=True)
        
        # build model
        params = {
        "input_shape": train_generator.dim, 
        "conv_layers":[(conv_filter_num, conv_filter_size, conv_layer_lambda)],
        "lstm_layers": [],
        "fc_layers":[(fc_neurons,fc_layer_lambda,dropout_rate),
                     (train_generator.n_classes,None,None)],
        "learning_rate":learning_rate
        }
         
        model = build_model(**params)
        model.summary()
        
        # Train
        model.fit_generator(generator=train_generator,
                  validation_data=cv_generator,
                  use_multiprocessing=True,
                  workers=4,
                  epochs=n_epoch,
                  class_weight=train_generator.class_weights,
                  callbacks=[stopping, reduce_lr, model_checkpoint],
                  verbose=1)
        
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


if __name__ == "__main__":
    bayes = BayesTrainer()
    optimizer = bayes.optimize()
    for i, res in enumerate(optimizer.res):
        print("Iteration {i}:\n\t{res}")
        
    