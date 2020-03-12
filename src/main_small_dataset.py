#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:40:37 2020

@author: danielyaeger
"""
import numpy as np
import pickle
import time
import os
import gc
from pathlib import Path
import tensorflow as tf
import datetime
from tensorflow import keras
from model_functions.model_functions import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from data_generators.data_generator_apnea_test import DataGeneratorApneaTest
import matplotlib.pyplot as plt
from collections import defaultdict

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

def test_wrapper_one_dir(path = '/Users/yaeger/Documents/datasets_a_b/raw_no_baseline_all', K_FOLD = 3,
                 out_path = '/Users/yaeger/Documents/datasets_a_b/raw_no_baseline_all'):
    if not isinstance(path, Path): path = Path(path)
    if not isinstance(out_path, Path): out_path = Path(out_path)
    IDs = [f.name.split('.')[0] for f in path.iterdir() if f.suffix == '.npy']
    bal_acc = np.zeros(len(IDs))
    for i, ID in enumerate(IDs):
        print(f'Processing {ID}')
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

########### Generators #################################################

class Main():
        def __init__(self, ID: str = 'XAXVDJYND80EZPY', K_FOLD: int = 5,
                          data_path: str = '/Users/danielyaeger/Documents/baselined_raw_no_rep_zero_no_upper'):
            self.ID = ID
            self.K_FOLD = K_FOLD
            self.data_path = Path(data_path)
            self.bal_acc = np.zeros(K_FOLD)


        def main(self):
            for k in range(self.K_FOLD):
                val = self.train(k)
                self.test(k,val)
                gc.collect()
            return self.bal_acc.mean()

        def train(self, k):
            print(f'ID: {self.ID}\t data_path: {str(self.data_path)}')
            print(f'\tFOLD = {k}\n\n\n\n')

            #assert data_path == '/floyd/input/data', f'Data path is {data_path}!'
            train_generator = DataGeneratorApneaTest(n_classes = 2,
                                            data_path = self.data_path,
                                            single_ID = self.ID,
                                            k_fold = self.K_FOLD,
                                            baseline_train = False,
                                            test_index = k,
                                            mode="train",
                                            context_samples=300)
            trainX,trainY = train_generator.get_data()

            cv_generator =  DataGeneratorApneaTest(n_classes = 2,
                                            data_path = self.data_path,
                                            mode="val",
                                            baseline_train = False,
                                            single_ID = self.ID,
                                            k_fold = self.K_FOLD,
                                            test_index = k,
                                            context_samples=300)
            val = cv_generator.get_data()




            ########### Make and print Model #############################################
            learning_rate = 1e-3

            stopping = EarlyStopping(patience=5)

            reduce_lr = ReduceLROnPlateau(factor=0.1,
                                            patience=8,
                                            min_lr=1e-6)

            model_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                                 monitor='loss',
                                                 save_best_only=True)

            params = {
            "input_shape": train_generator.dim,
            "lstm_layers":[(128,True,0.1),(128,False,0.1)],
            "fc_layers":[128,64,train_generator.n_classes],
            "learning_rate":learning_rate
            }

            model = build_model(**params)
            model.summary()


            ########## Train Model #######################################################
            if len(train_generator.class_weights) == 1:
                model.fit(x = trainX,
                y = trainY,
                batch_size = 128,
                epochs = 20,
                callbacks=[stopping, reduce_lr, model_checkpoint], #, metric],
                verbose = 2,
                validation_data = val,
                shuffle = True,
                use_multiprocessing=True,
                workers=4)

            else:
                model.fit(x = trainX,
                y = trainY,
                batch_size = 128,
                epochs = 20,
                callbacks=[stopping, reduce_lr, model_checkpoint], #, metric],
                verbose = 2,
                validation_data = val,
                shuffle = True,
                use_multiprocessing=True,
                class_weight=train_generator.class_weights,
                workers=4)

            return val

        def test(self,k,val):

            ########## Make Predictions ###################################################
            # Roll back to best model
            model = load_model('model.hdf5')
           # print('\tLoaded best model')

            #Calculate balanced accuracy
            X,y_true = val
            y_pred = model.predict(X)
            y_true = y_true[0:len(y_pred)]
            assert len(y_pred) == len(y_true),f'Length y_pred: {len(y_pred)}\t Length y_true: {len(y_true)}'
            try:
                self.bal_acc[k] = balanced_accuracy_score(y_true.argmax(-1), y_pred.argmax(-1))
            except:
                self.bal_acc[k] = np.nan
            print(f"\tFor ID: {self.ID}\tfold {k}: balanced accuracy: {self.bal_acc[k]}")

if __name__ == "__main__":
    test_wrapper_one_dir(path = '/Users/yaeger/Documents/datasets_a_b/raw_no_baseline_all', K_FOLD = 3,
                 out_path = '/Users/yaeger/Documents/datasets_a_b/raw_no_baseline_all')
