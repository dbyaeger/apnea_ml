#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:20:03 2020

@author: danielyaeger
"""
from pathlib import Path
from model_functions.model_functions import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score
from bayes_opt import BayesianOptimization
from data_generators.data_generator_apnea_ID_batch import DataGeneratorApneaIDBatch
from data_generators.data_generator_apnea import DataGeneratorApnea



class BayesTrainer():
    """Wrapper class to optimize a model using Bayesian Optimization
    package
    """
    def __init__(self, data_path: str, model_path: str, pbounds: dict = {'dropout_rate': (0.1,0.9),
                 'conv_layer_lambda': (1e-4,1), 'conv_filter_size': (10,200),
                 'fc_neurons': (32,512),'fc_layer_lambda': (1e-4,1)},
                 init_points: int = 10, n_iter: int = 10):
        
        if not isinstance(data_path, Path): data_path = Path(data_path)
        self.data_path = data_path
        
        if not isinstance(model_path, Path): model_path = Path(model_path)
        self.model_path = model_path
        
        if not self.model_path.is_dir():
            self.model_path.mkdir()
            
        self.pbounds = pbounds
        self.init_points = init_points
        self.n_iter = n_iter
        self.iterations = -1
        
        
    def fit_with(self,dropout_rate, conv_layer_lambda, conv_filter_size, fc_neurons, fc_layer_lambda):
        """Builds a 1-conv layer, 2-dense layer neural net with specified 
        """
        # set iterations variable for model saving
        self.iterations += 1
            
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
                                    shuffle=True)
        
        # Set parameters                            
        learning_rate = 1e-3
        n_epoch =20
        stopping = EarlyStopping(patience=5)

        reduce_lr = ReduceLROnPlateau(factor=0.1,
                                        patience=8,
                                        min_lr=1e-6)
        model_path = f'/Users/danielyaeger/Documents/Modules/apnea_ml/src/models/1_conv_2_fc/model_{self.iterations}.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=model_path, 
                                             monitor='loss', 
                                             save_best_only=True)
        
        # build model
        params = {
        "input_shape": train_generator.dim, 
        "conv_layers":[(64, int(round(conv_filter_size)), conv_layer_lambda)],
        "lstm_layers": [],
        "fc_layers":[(int(round(fc_neurons)),dropout_rate,fc_layer_lambda),
                     (train_generator.n_classes,None,None)],
        "learning_rate":learning_rate
        }
         
        model = build_model(**params)
        
        # Train
        model.fit(train_generator,
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
                                    self.data_path = data_path,
                                    batch_size = 128,
                                    mode="cv",
                                    context_samples=300,
                                    shuffle=False)
        y_pred = best_model.predict_generator(cv_generator)
        y_true = cv_generator.labels[:len(y_pred)]
        
        score = balanced_accuracy_score(y_true, y_pred.argmax(-1))
        
        print(f'Balanced accuracy score: {score}')
        
        return score
    
    def optimize(self):
        """Wrapper function for BayesianOptimization
        """
        optimizer = BayesianOptimization(f=self.fit_with, pbounds=self.pbounds,
                                         verbose=2,random_state=1)
        
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        
        return optimizer

if __name__ == "__main__":
    bayes = BayesTrainer()
    optimizer = bayes.optimize()
    for i, res in enumerate(optimizer.res):
        print("Iteration {i}:\n\t{res}")
        
    