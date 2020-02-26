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
from tensorflow.keras import Sequential, Input, Model, callbacks, layers
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Add, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K 
#from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from data_generator_apnea_test import DataGeneratorApneaTest
import matplotlib.pyplot as plt
from collections import defaultdict

######################## Create folder to store results ######################
# print('Make results folder')
# out_path = Path('results')
# if not out_path.is_dir(): out_path.mkdir()
    
######### Functions for building network #####################################
def _add_conv_layer(layer, n_filters, filter_size, stride=1,
                    input_shape=None, activation="relu", 
                    batch_norm=True, pool=True, conv2D=False):
    """
    Add a conv layer to network architecture
    INPUTS:
        layer: input to layer
        n_filters: nunber of filters in current layer
        filter_size: filter size for current layer
        stride: stride for current layer
        input_shape: input shape to current layer, only needed for first layer
        activation: activation function for current layer
        batch_norm: will current layer use batch normalization
        pool: will current layer use max pooling
        conv2d: is current layer going to use 2d conv (True) or 1d (False)
    OUTPUTS:
        conv layer
    """
    if conv2D:
        conv = Conv2D
        pool = MaxPool2D
    else:
        conv = Conv1D
        pool = MaxPool1D
    if input_shape:
        layer = conv(n_filters, filter_size, stride=stride,
                       input_shape=input_shape)(layer)
    else:
        layer = conv(n_filters, filter_size, stride)(layer)
    layer = Activation(activation)(layer)
    if batch_norm:
        layer = BatchNormalization()(layer)
    if pool:
        layer = pool(2)(layer)
    return layer


def _add_dense_layer(layer, n_out, activation="relu", dropout_p=None):
    """
    Add a dense layer to network architecture
    INPUTS:
        layer: input to layer
        n_out: number of output neurons
        activation: activation function for current layer
        dropout_p: retain probability for current layer (if None -> no dropout)
    OUTPUTS:
        dense layer
    """
    layer = Dense(n_out)(layer)
    if activation:
        layer = Activation(activation)(layer)
    if dropout_p:
        layer = Dropout(dropout_p)(layer)
    return layer

def build_model(**params):
    """
    Function for build a network based on the specified input params
    INPUTS:
        params = {"conv_layers":[(input_size_i, output_size_i, stride_i), ...],
                  "fc_layers": [output_size_i, ...],
                  "input_shape":(n, m, ...),
                  "callbacks":[callback_i, ...],
                  "learning_rate":1e-3,
                  ...}
    OUTPUTS:
        compiled network
    """
    conv_layers = params["conv_layers"]
    fc_layers = params["fc_layers"]
    input_ = Input(shape=params["input_shape"])
    if params.get("learning_rate"):
        learning_rate = params["learning_rate"]
    else:
        learning_rate = 1e-3
    out = input_
    for i, p in enumerate(conv_layers):
        out = _add_conv_layer(out, *p)
    out = Flatten()(out)
    for i, n in enumerate(fc_layers):
        if i < len(fc_layers) - 1:
            out = _add_dense_layer(out, n, activation="relu", dropout_p=0.5)
        else:
            out = _add_dense_layer(out, n, activation="softmax")
    optim = Adam(learning_rate)
    model = Model(input_, out)
    model.compile(optimizer=optim, 
                  loss="categorical_crossentropy",
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model

class BalancedAccuracy(Callback):
    def on_train_begin(self, logs={}):
        self.val_balanced_accuracy = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.model.validation_data[0]))
        val_targ = self.model.validation_data[1]
        _bal_accuracy = balanced_accuracy_score(val_targ.argmax(-1), val_predict.argmax(-1))
        self.val_balanced_accuracy.append(_bal_accuracy)
        print(f'val_balanced_accuracy: {_bal_accuracy}')
        return
         

    


########### Model Parameters #################################################
n_epoch = 20
learning_rate = 1e-3

stopping = callbacks.EarlyStopping(patience=10)

reduce_lr = callbacks.ReduceLROnPlateau(factor=0.1,
                                        patience=8,
                                        min_lr=1e-6)

model_checkpoint = callbacks.ModelCheckpoint(filepath='model.hdf5', 
                                             monitor='loss', 
                                             save_best_only=True)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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
                                            test_index = k,
                                            mode="train",
                                            context_samples=300)
            trainX,trainY = train_generator.get_data()
                    
            cv_generator =  DataGeneratorApneaTest(n_classes = 2,
                                            data_path = self.data_path,
                                            mode="val",
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
            "conv_layers":[(64, 100)],
            "fc_layers":[128,train_generator.n_classes],
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

        
        
   