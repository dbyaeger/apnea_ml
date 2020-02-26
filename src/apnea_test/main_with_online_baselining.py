#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:01:15 2020

@author: danielyaeger
"""
from src.utils.baseliner import set_baseline
from pathlib import Path
import pickle
import gc
import numpy as np
from model_functions import build_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score
from data_generator_apnea_test import DataGeneratorApneaTest

class LossMonitor():
    
    def __init__(self, patience: int = 5):
        self.patience = patience
    
    def stop_training(self,loss):
        """
          Inputs: Loss, list of scalars
              
          Returns False if:
            a) the length of the loss list is less than patience.
            b) the next consecutive patience entries do not have greater loss 
               than the ith entry.
           Returns True:
               if all of the next consecutive patience entries have greater loss 
               than the ith entry
        """
        if len(loss) <= self.patience:
            return False
        else:
            for i in range(0, len(loss)-self.patience):
                for j in range(self.patience+1):
                    if loss[i+j] < loss[i]:
                        return False
        return True
    

class OnlineBaseliner():
    def __init__(self, model, x, channel_list, set_baseline = set_baseline, 
                 cut_offs = (0.01,0.99), stepsize = 10, sampling_rate =10):
        
        self.model = model
        self.x = x
        self.set_baseline = set_baseline
        self.stepsize = stepsize
        self.channel_list = channel_list
        self.labels = np.zeros(len(x))
        self.sampling_rate = sampling_rate
        self.first_pass()
        
        
        
    def first_pass(self):
        """Performs a first pass. First baselines data naively, and then if an
        apneic/hypopneic event is predicted, re-baselines the data."""
        for index in range(self.stepsize,(len(self.x)+self.stepsize),self.stepsize):
            temp_x = self.baseliner(index)
            self.labels[index-self.stepsize:index] = self.model.predict(temp_x).argmax(-1)
            if self.labels[index-self.stepsize:index].sum() > 0:
                self.x[index-self.stepsize:index,:] = self.baseliner(index)
    
    def get_loss(self,y):
        "Returns the loss for the online-baselined data"
        return self.model.evaluate(self.x,y)
    
    def get_pred(self):
        "Returns predictions for the online-baselined data"
        return self.model.predict(self.x)
        
    
    def baseliner(self, index):
        "Baselines data"
        temp_x = self.x[index-self.stepsize:index,:]
        for channel in self.channel_list:
            print(channel)
            if channel == 'SpO2':
                baseline = self.set_baseline(data=self.x[:,self.channel_list.index(channel)],
                                                                         labels = self.labels,
                                                                         baseline_type = 'quantile',
                                                                         sampling_rate=self.sampling_rate,
                                                                         quantile=0.95,
                                                                         baseline_length=120,
                                                                         step_size=self.stepsize)
                temp_x[:,self.channel_list.index(channel)] = temp_x[:,self.channel_list.index(channel)]//baseline[0]
                
            elif channel != 'ECG':
                baseline = self.set_baseline(data=self.x[:,self.channel_list.index(channel)],
                                                                         labels = self.labels,
                                                                         baseline_type = 'min_max',
                                                                         sampling_rate=self.sampling_rate,
                                                                         cut_offs = self.cut_offs,
                                                                         baseline_length=120,
                                                                         step_size=self.stepsize)
                temp_x[:,self.channel_list.index(channel)] = (temp_x[:,self.channel_list.index(channel)] - baseline[0])/(baseline[1]-baseline[0])
                
        return temp_x
        

class Main():
    def __init__(self, ID: str = 'XAXVDJYND80EZPY', K_FOLD: int = 5, 
                      data_path: str = '/Users/danielyaeger/Documents/baselined_raw_no_rep_zero_no_upper'):
        self.ID = ID
        self.K_FOLD = K_FOLD
        self.data_path = Path(data_path)
        self.bal_acc = np.zeros(K_FOLD)
        with self.data_path.joinpath('channel_list.p').open('rb') as fc:
            self.channel_list = pickle.load(fc)
        
    
    def main(self):
        for k in range(self.K_FOLD):
            val, best_model = self.train(k)
            self.test(k,val,best_model)
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
        valX, valY = val
        
        
        
        
        ########### Make and print Model #############################################
        learning_rate = 1e-3


        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=8, min_lr=1e-6)

        params = {
        "input_shape": train_generator.dim, 
        "conv_layers":[(64, 100)],
        "fc_layers":[128,train_generator.n_classes],
        "learning_rate":learning_rate
        }
        
        model = build_model(**params)
        #model.summary()
        
        
        ########## Train Model #######################################################
        epoch = 0
        loss = []
        lm = LossMonitor()
        
        while epoch < 20:
            
            print(f'\t\tTRAINING: EPOCH {epoch}')
            
            model_checkpoint = ModelCheckpoint(filepath= f'model_{epoch}.hdf5', 
                                             save_best_only=False)
            
            best_model = f'model_{epoch}.hdf5'
            
            model.fit(x = trainX,
            y = trainY,
            batch_size = 128,
            epochs = 1,
            callbacks=[reduce_lr, model_checkpoint],
            verbose = 2,
            shuffle = True,
            use_multiprocessing=True,
            workers=4)
            
            print(f'\t\tVALIDATING')
            
            ob = OnlineBaseliner(model, valX, self.channel_list)
            loss.append(ob.get_loss(valY))
            
            print(f'\t\tVALIDATION LOSS: {loss[-1]}')
            
            if lm.stop_training(loss):
                best_model = f'model_{np.argmin(loss)}.hdf5'
                break
            
            epoch += 1
        
        # Return the best model
        model = load_model(best_model)
        
        return val, model

            
    def test(self,k,val,model):
    
        ########## Make Predictions ###################################################
      
        valX, valY = val
        ob = OnlineBaseliner(model, valX, self.channel_list)
        y_pred = ob.get_pred()
        y_true = valY[0:len(y_pred)]
        assert len(y_pred) == len(y_true),f'Length y_pred: {len(y_pred)}\t Length y_true: {len(y_true)}'
        try:
            self.bal_acc[k] = balanced_accuracy_score(y_true.argmax(-1), y_pred.argmax(-1))
        except:
            self.bal_acc[k] = np.nan
        print(f"\tFor ID: {self.ID}\tfold {k}: balanced accuracy: {self.bal_acc[k]}")
