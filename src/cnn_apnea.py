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
#from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from data_generator_apnea import DataGeneratorApnea
from model_functions.model_functions import build_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import pyximport
pyximport.install()
from viterbi import post_process

######################## Create folder to store results ######################
# print('Make results folder')
# out_path = Path('results')
# if not out_path.is_dir(): out_path.mkdir()
    

########### Generators #################################################
print('Making generators')
data_path = Path('/floyd/input/data')

######### Normalize ####################################################
with data_path.joinpath('ID_partitions.p').open('rb') as fid: IDs = pickle.load(fid)


#assert data_path == '/floyd/input/data', f'Data path is {data_path}!'
train_generator = DataGeneratorApnea(n_classes = 2,
                                data_path = data_path,
                                batch_size = 128,
                                mode="train",
                                context_samples=300,
                                load_all_data=True,
                                shuffle = True,
                                desired_number_of_samples = 2.1e6)

print(f'train_generator number of observations: {len(train_generator)}')
print(f'train_generator dimension: {train_generator.dim}')
print(f'train_generator class weights: {train_generator.class_weights}')

cv_generator =  DataGeneratorApnea(n_classes = 2,
                                data_path = data_path,
                                batch_size = 128,
                                mode="cv",
                                context_samples=300,
                                load_all_data=True,
                                shuffle = True)
print(f'cv_generator number of observations: {len(cv_generator)}')



########### Model Parameters #################################################
n_epoch = 200
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


params = {
    "input_shape": train_generator.dim, 
    "conv_layers":[(64, 100)],
    "fc_layers":[128,train_generator.n_classes],
    "learning_rate":learning_rate
    }

########### Make and print Model #############################################
model = build_model(**params)
model.summary()

########## Train Model #######################################################
history = model.fit_generator(train_generator,
                              validation_data=cv_generator,
                              use_multiprocessing=True,
                              workers=4,
                              epochs=n_epoch,
                              class_weight=train_generator.class_weights,
                              callbacks=[stopping, reduce_lr, model_checkpoint])#, tensorboard_callback,FloydhubTrainigMetricsCallback()])

#history_dict = {"train_loss": history.history["loss"],
#                "val_loss": history.history["val_loss"],
#                "train_cat_acc": history.history["categorical_accuracy"],
#                "val_cat_acc": history.history["val_categorical_accuracy"]}
#
#with open('history_apnea.p','wb') as fh: pickle.dump(history_dict,fh)

########## Make Predictions ###################################################
# Roll back to best model
model = load_model('model.hdf5')
print('Loaded best model')
model.summary()

def sleeper_metrics(model, list_IDs, save_name = 'cv_metrics', data_path = Path('/floyd/input/data')):
    """
    Function to make predictions on all windows in test set 
    and serialize output and ground truth into a single dict for all IDs
    """
    predictions = {}
    for i, ID in enumerate(list_IDs):
        print(f'ID: {ID}')
        assert ID in list_IDs, f'{ID} not in list_IDs:\n{list_IDs}'
        predictions[ID] = {}
        dg =  DataGeneratorApnea(
                                data_path = data_path,
                                batch_size = 64,
                                n_classes = 2,
                                mode='test',
                                context_samples=300,
                                load_all_data=True,
                                shuffle = False,
                                single_ID = ID)
        y_pred = model.predict_generator(dg)
        y_true = dg.labels[:len(y_pred)]
        print(f'Length of y_pred: len(y_pred)')
        print(f'Length of y_pred: len(y_true)')
        assert len(y_pred) == len(y_true),f'Length y_pred: {len(y_pred)}\t Length y_true: {len(y_true)}'
        predictions[ID]["targets"] = y_true
        predictions[ID]["predictions"] = y_pred
        predictions[ID]["evaluation"] = {}
        try:
            predictions[ID]["evaluation"]["balanced_accuracy"] = balanced_accuracy_score(y_true,
                                                                                         y_pred.argmax(-1))
        except:
            predictions[ID]["evaluation"]["balanced_accuracy"] = np.nan

        predictions[ID]["evaluation"]["confusion_matrix"] = confusion_matrix(y_true,
                                                                             y_pred.argmax(-1))
        predictions[ID]["evaluation"]["classification_report"] = classification_report(y_true,
                                                                             y_pred.argmax(-1))
        print(f'\tbal. acc: {predictions[ID]["evaluation"]["balanced_accuracy"]}')
        
    with open(save_name,'wb') as fh: pickle.dump(predictions,fh)


def overall_metrics(save_name = 'cv_overall', data_path = 'cv_metrics'):
    """Calculates overall metrics for data and reports and saves them
    """
    out = {}
    if not isinstance(data_path, Path): data_path = Path(data_path)
    with data_path.open('rb') as fin: data = pickle.load(fin)
    
    y_true = []
    y_pred = []
    for ID in data:
        y_true.extend(data[ID]['targets'])
        try:
            y_pred.extend(data[ID]['predictions'].argmax(-1))
        except:
            y_pred.extend(data[ID]['predictions'])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f'Shape of y_pred: {y_pred.shape}')
    print(f'Shape of y_pred: {y_true.shape}')
    assert len(y_pred) == len(y_true),f'Length y_pred: {len(y_pred)}\t Length y_true: {len(y_true)}'
    try:
        out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    except:
        out["balanced_accuracy"] = np.nan
        
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    out["classification_report"] = classification_report(y_true, y_pred)
    print(f'\tbal. acc: {out["balanced_accuracy"]}')
        
    with open(save_name, 'wb') as fh: pickle.dump(out,fh)

def viterbi_wrapper(
        save_name,
        path_to_probabilities,
        path_to_transition_matrix = '/floyd/home/apnea_hypopnea_transition_matrix'):
    
    smooth_output = {}
    
    with open(path_to_transition_matrix,'rb') as fi:
        transition_mat = pickle.load(fi)
    
    with path_to_probabilities.open('rb') as fh:
        data = pickle.load(fh)
    
    for ID in data:
        print(f'Smoothing {ID}')
        smooth_output[ID] = {}
        smooth_output[ID]['targets'] = data[ID]['targets']
        smooth_output[ID]['predictions'] = post_process(data[ID]['predictions'], transition_mat)
    
    with open(save_name, 'wb') as fh: pickle.dump(smooth_output,fh)

def simple_sleeper_metrics(save_name, data_path):
    out = {}
    with data_path.open('rb') as fh:
        data = pickle.load(fh)
    
    for ID in data:
        out[ID] = {}
        try:
            out[ID]["balanced_accuracy"] = balanced_accuracy_score(data[ID]['targets'], 
                                                               data[ID]['predictions'])
        except:
            out[ID]["balanced_accuracy"] = np.nan
       
        out[ID]["confusion_matrix"] = confusion_matrix(data[ID]['targets'], 
                                                               data[ID]['predictions'])
        out[ID]["classification_report"] = classification_report(data[ID]['targets'], 
                                                               data[ID]['predictions'])
        
    with open(save_name, 'wb') as fh: pickle.dump(out,fh)
    
        
        
            

sleeper_metrics(model = model, list_IDs = IDs['val'], save_name = 'cv_metrics.p', data_path = Path('/floyd/input/data'))
sleeper_metrics(model = model, list_IDs = IDs['test'], save_name = 'test_metrics.p', data_path = Path('/floyd/input/data'))
data_path = Path('/floyd/home')
overall_metrics(save_name = 'cv_overall_metrics.p', data_path = data_path.joinpath('cv_metrics.p'))
overall_metrics(save_name = 'test_overall_metrics.p', data_path = data_path.joinpath('test_metrics.p'))
viterbi_wrapper(save_name = 'smoothed_cv_output.p', path_to_probabilities = data_path.joinpath('cv_metrics.p'))
viterbi_wrapper(save_name = 'smoothed_test_output.p', path_to_probabilities = data_path.joinpath('test_metrics.p'))
overall_metrics(save_name = 'cv_overall_smoothed_metrics.p', data_path = Path('/floyd/home/smoothed_cv_output.p'))
overall_metrics(save_name = 'test_overall_smoothed_metrics.p', data_path = Path('/floyd/home/smoothed_test_output.p'))
simple_sleeper_metrics(save_name = 'cv_sleeper_smoothed.p', data_path =Path('/floyd/home/smoothed_cv_output.p'))
simple_sleeper_metrics(save_name = 'test_sleeper_smoothed.p', data_path =Path('/floyd/home/smoothed_test_output.p'))
