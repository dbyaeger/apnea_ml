#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:01:15 2020

@author: danielyaeger
"""
from utils.baseliner import set_baseline
from pathlib import Path
import pickle
import gc
import numpy as np
from run_utils.loss_monitor import LossMonitor
from run_utils.online_baseliner import OnlineBaseliner
from model_functions import build_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score
from data_generator_apnea_test import DataGeneratorApneaTest

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
        tester = Main(ID=ID, K_FOLD=K_FOLD, data_path = str(path))
        bal_acc_for_ID = tester.main()
        bal_acc[i] = bal_acc_for_ID
        print(f'MEAN BALANCED ACCURACY FOR {ID}: {bal_acc_for_ID}')
        del tester
        gc.collect()
    print(f'OVERALL MEAN BALANCED ACCURACY: {np.mean(bal_acc)}')         
    with out_path.joinpath('results_25_75').open('wb') as fh:
        pickle.dump(bal_acc,fh)
    return bal_acc

class Main():
    def __init__(self, ID: str = 'XAXVDJYND80EZPY', K_FOLD: int = 3, 
                      data_path: str = '/Users/danielyaeger/Documents/baselined_raw_no_rep_zero_no_upper'):
        self.ID = ID
        self.K_FOLD = K_FOLD
        self.data_path = Path(data_path)
        self.bal_acc = np.zeros(K_FOLD)
        with self.data_path.joinpath('channel_list.p').open('rb') as fc:
            self.channel_list = pickle.load(fc)
        
    
    def main(self):
        for k in range(self.K_FOLD):
            y_pred, y_true = self.train(k)
            self.test(k,y_pred, y_true)
            gc.collect()
        return self.bal_acc.mean()
    
    def train(self, k):
        print(f'ID: {self.ID}\t data_path: {str(self.data_path)}')
        print(f'\tFOLD = {k}\n\n\n\n')
        
        label_path = Path('/Users/danielyaeger/Documents/Modules/apnea_ml/src/predictions')
        
        #assert data_path == '/floyd/input/data', f'Data path is {data_path}!'
        train_generator = DataGeneratorApneaTest(n_classes = 2,
                                        data_path = self.data_path,
                                        single_ID = self.ID,
                                        k_fold = self.K_FOLD, 
                                        test_index = k,
                                        mode="train",
                                        context_samples=300)
        trainX,trainY = train_generator.get_data()
 
        ########### Make and print Model #############################################
        learning_rate = 1e-3


        #reduce_lr = ReduceLROnPlateau(factor=0.1, patience=8, min_lr=1e-6)

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
            
            best_model = f'./models/model_{epoch}.hdf5'
            best_labels = label_path.joinpath(f'{epoch}.npy')
            
            model_checkpoint = ModelCheckpoint(filepath= best_model, 
                                             save_best_only=False)
            
            
            model.fit(x = trainX,
            y = trainY,
            batch_size = 128,
            epochs = 1,
            callbacks=[model_checkpoint],
            verbose = 1,
            shuffle = True,
            use_multiprocessing=True,
            workers=4)
            
            print(f'\t\tVALIDATING')
            
            cv_generator =  DataGeneratorApneaTest(n_classes = 2,
                                        data_path = self.data_path,
                                        mode="val",
                                        single_ID = self.ID,
                                        k_fold = self.K_FOLD, 
                                        test_index = k,
                                        batch_size = 20,
                                        context_samples=300)
            
            ob = OnlineBaseliner(model, cv_generator, epoch, label_path)
            loss.append(ob.get_loss()[0])            
            
            if lm.stop_training(loss):
                best_labels = label_path.joinpath(f'{np.argmin(loss)}.npy')
                break
            
            epoch += 1
        
        # Return the best model
        y_pred = np.load(str(best_labels))
        
        print(f'\t\tBest validation loss on epoch {np.argmin(loss)}')
        
        y_true = cv_generator.labels
        
        return y_pred, y_true

            
    def test(self,k,y_pred, y_true):
    
        ########## Make Predictions ###################################################
        assert len(y_pred) == len(y_true),f'Length y_pred: {len(y_pred)}\t Length y_true: {len(y_true)}'
        try:
            self.bal_acc[k] = balanced_accuracy_score(y_true, y_pred)
        except:
            self.bal_acc[k] = np.nan
        print(f"\tFor ID: {self.ID}\tfold {k}: balanced accuracy: {self.bal_acc[k]}")

if __name__ == "__main__":        
    results = test_wrapper_one_dir()
    
