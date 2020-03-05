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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from data_generators.data_generator_apnea_test import DataGeneratorApneaTest
import matplotlib.pyplot as plt
from collections import defaultdict


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

            stopping = callbacks.EarlyStopping(patience=5)

            reduce_lr = callbacks.ReduceLROnPlateau(factor=0.1,
                                            patience=8,
                                            min_lr=1e-6)

            model_checkpoint = callbacks.ModelCheckpoint(filepath='model.hdf5',
                                                 monitor='loss',
                                                 save_best_only=True)

            params = {
            "input_shape": train_generator.dim,
            "lstm_layers":[(128,True,0.1),(128,False,0.1)],
            "fc_layers":[128,64,train_generator.n_classes],
            "learning_rate":learning_rate
            }

            metric = BalancedAccuracy()
            model = build_model(**params)
            #model.summary()


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
