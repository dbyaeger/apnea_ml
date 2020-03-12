#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:28:42 2019

@author: danielyaeger
"""
from pathlib import Path
import numpy as np
import pickle
from utils.baseliner import baseline, set_baseline

class DataGeneratorApneaTest():
    'Data Generator for TensorFlow'
    def __init__(self, data_path: str = '/Users/danielyaeger/Documents/raw_no_baseline',
                 sampling_rate: int = 10, n_classes = 2,
                 context_samples: int = 300, batch_size = 10,
                 single_ID = 'XAXVDJYND80EZPY',
                 k_fold = 5, test_index = 0, mode = 'train',
                 baseline_train = False, cut_offs: (float,float) = (0.25,0.75)):


        if type(data_path) == str: data_path = Path(data_path)
        self.data_path = data_path
        assert self.data_path.is_dir(), f"{data_path} is not a valid path"
        
        with self.data_path.joinpath('channel_list.p').open('rb') as fl:
            channel_list = pickle.load(fl)
        
        self.channel_list = channel_list
        
        self.n_channels = len(self.channel_list)
        
        
        with self.data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
            self.targets = pickle.load(ta)
        
        assert n_classes in [2,3], 'n_classes must be 2 or 3!'
        self.n_classes = n_classes
        if self.n_classes == 2:
            # Convert targets from thrinary to binary
            for ID in self.targets:
                idx = np.nonzero(self.targets[ID] > 1)[0]
                if len(idx) > 0:
                    self.targets[ID][idx] = 1     

            
        self.single_ID = single_ID
        assert 0 <= test_index <= (k_fold -1), "test index not between 0 and k_fold - 1!"
        
        self.mode = mode
        assert mode in ['val','test','cv','train'], f'Mode is {mode}!'
        
        
        # Get data to split up
        variables = np.load(str(self.data_path.joinpath(single_ID + '.npy')))
        length = len(self.targets[single_ID])
        fold_size = length//k_fold
            
        # Make dictionaries to store data
        self.data = {i: variables[i*fold_size:(i+1)*fold_size,:] for i in range(k_fold)}
        self.targets = {i: self.targets[single_ID][i*fold_size:(i+1)*fold_size] for i in range(k_fold)}
        
        self.context_samples = context_samples
        self.fs = sampling_rate
        self.cut_offs = cut_offs
        self.batch_size = batch_size
        self.test_index = test_index
        
        if mode == 'train':
            # Delete test_index if mode is not 'val' or 'train'
            del self.data[test_index]
            del self.targets[test_index]
            if baseline_train:                                                                 
                self.baseline_all_data()                                                                     
   
        elif mode in ['val','cv','test']:
            self.data = {test_index: self.data[test_index]}
            self.copy = {test_index: self.data[test_index].copy()}
            self.targets = {test_index: self.targets[test_index]}
            assert len(list(self.data.keys())) == len(list(self.targets.keys())) == 1, f'self.targets keys: {self.targets.keys()}'
        
        if self.n_classes == 3:
            self.label_dict = {'0': 0, 'H': 1, 'A': 2}
       
        elif self.n_classes == 2:
            self.label_dict = {'0': 0, 'A/H': 1}

        self.inverse_label_dict = {self.label_dict[label]: label for label in self.label_dict}
                
        self.dim = (2*context_samples + 1, self.n_channels)
        
        self.make_samples()
        
    def baseline_all_data(self, labels = None):
        """Baselines training data"""       
        for fold in self.data:
            if labels is None:
                labels = self.targets[fold]
            for channel in self.channel_list:
                if channel == 'SpO2':
                    self.data[fold][:,self.channel_list.index(channel)] = baseline(data=self.data[fold][:,self.channel_list.index(channel)],
                                                                             labels = labels,
                                                                             baseline_type = 'quantile',
                                                                             sampling_rate=10,
                                                                             quantile=0.95,
                                                                             baseline_length=120,
                                                                             step_size=10)
                elif channel != 'ECG':
                    self.data[fold][:,self.channel_list.index(channel)] = baseline(data=self.data[fold][:,self.channel_list.index(channel)],
                                                                             labels = labels,
                                                                             baseline_type = 'min_max',
                                                                             sampling_rate=10,
                                                                             cut_offs = self.cut_offs,
                                                                             baseline_length=120,
                                                                             step_size=10)


    def make_samples(self):
        """If mode is set to train, creates samples by randoming sampling each
        label in proportion to the total amount of samples for each sleeper"""
        # Randomly sample minority classes to bring count up to level of majority class
        self.samples = []
       
        if self.mode == 'train':
            
            for ID in self.targets:
                # Add A/H samples
                indexes = np.nonzero(self.targets[ID] > 0)[0]
                
                
                if len(indexes) > 0:
                    
                    samp_list = [(ID, idx, int(self.targets[ID][idx])) for idx in indexes]
                    
                    # Add equal number of zero samples
                    indexes = np.nonzero(self.targets[ID] == 0)[0]
                    np.random.shuffle(indexes)
                    samp_list.extend([(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:len(samp_list)]])
                
                else:
                    # arbitrarily add 10% of data
                    indexes = np.nonzero(self.targets[ID] == 0)[0]
                    samp_list= [(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:len(indexes)//10]]

                # Add samples to master list
                self.samples.extend(samp_list)
                
                assert len(self.samples) > 0, "self.samples has length of zero!"
                
        else:
            for ID in self.targets:
                for idx in range(self.targets[ID].shape[0]):
                    self.samples.append((ID, idx, int(self.targets[ID][idx])))
  
    @property
    def labels(self):
        return [x[-1] for x in self.samples]
    
    @property
    def class_weights(self):
        "Calculates class_weights"
        class_counts = {}
        # Get counts
        for label in self.label_dict:
            class_counts[self.label_dict[label]] = sum([1 for x in self.samples if x[-1] == self.label_dict[label]])
        # Normalize to max_count
        class_weights = {}
        for label in class_counts:
            try:
                class_weights[label] = max(list(class_counts.values()))/class_counts[label]
            except:
                continue
        return class_weights
    
    
    def get_data(self, usecopy = False):
        """Returns all data"""
        X = np.zeros((len(self.samples), *self.dim), dtype=np.float32)
        y = np.zeros((len(self.samples), self.n_classes))
        for i, item in enumerate(self.samples):
            ID, idx, label = item
            features, label = self._sample(ID, idx, label, usecopy)
            X[i], y[i] = features, label

        return X, y
    
    def refresh_copy(self, index):
        "Restores copy of data for specified indices of batch"
        samples = self.samples[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the index of the samples
        idxs = [item[1] for item in samples]
        
        # Find start index
        start_idx = min(idxs)
        start_idx = max(0, start_idx - self.context_samples)
         
        # Find end index
        end_idx = max(idxs)
        end_idx = min(len(self.samples), end_idx + self.context_samples)
        
        self.copy[self.test_index][start_idx:end_idx+1] = self.data[self.test_index][start_idx:end_idx+1].copy()
    
    def __getitem__(self,index, usecopy=True):
        """Returns one batch of data"""
        X = np.zeros((self.batch_size, *self.dim), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_classes))
        samples = self.samples[index*self.batch_size:(index+1)*self.batch_size]
        for i, (ID, idx, label) in enumerate(samples):
            features, label = self._sample(ID, idx, label, usecopy)
            X[i], y[i] = features, label
        return X, y
                      
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.samples) // self.batch_size
    
    def __baseline_item__(self,index,labels,usecopy=True):
        """Baselines one batch of data. Assumes all data has the same ID"""
        # Get samples. Assumes samples are ordered 
        samples = self.samples[index*self.batch_size:(index+1)*self.batch_size]
        
        # Get the index of the samples
        idxs = [item[1] for item in samples]
        
        # Find start index
        start_idx = min(idxs)
        start_idx = max(0, start_idx - self.context_samples)
         
        # Find end index
        end_idx = max(idxs)
        end_idx = min(len(labels), end_idx + self.context_samples)
        
        if usecopy:
            data = self.copy[self.test_index]
        else:
            data = self.data[self.test_index]
        
        # Set baseline
        for channel in self.channel_list:
            if channel == 'SpO2':
                baseline = set_baseline(data=data[:,self.channel_list.index(channel)],
                                        index = end_idx,
                                        labels = labels,
                                        baseline_type = 'quantile',
                                        sampling_rate=self.fs,
                                        quantile=0.95,
                                        baseline_length=120)
                data[start_idx:end_idx+1,self.channel_list.index(channel)] = data[start_idx:end_idx+1,self.channel_list.index(channel)]/baseline[0]
                
            elif channel != 'ECG':
                baseline = set_baseline(data=data[:,self.channel_list.index(channel)],
                                        index = end_idx,
                                        labels = labels,
                                        baseline_type = 'min_max',
                                        sampling_rate=self.fs,
                                        cut_offs=self.cut_offs,
                                        baseline_length=120)
                data[start_idx:end_idx+1,self.channel_list.index(channel)] = (data[start_idx:end_idx+1,self.channel_list.index(channel)] - baseline[0])/(baseline[1]-baseline[0])
                
        
        
        
    
    def _sample(self, ID, center_idx, label, usecopy):
        """ Returns random or weighted random sampled from ID.
        """
        # Check if epoch in sleep study
        assert 0 <= center_idx <= len(self.targets[ID])-1, f"Index: {center_idx} not in apnea_hypopnea_targets.p for {ID}!"
        
        # Initialize padding variable
        left_pad, right_pad = None, None
        
        # Zero-pad on left if necessary
        start_idx = int(center_idx - self.context_samples)-1
        if start_idx < 0:
            left_pad = np.zeros((-start_idx,self.n_channels))
            start_idx = 0

        # Zero-pad on right if necessary
        end_idx = int(center_idx + self.context_samples)
        end_sig_idx = len(self.targets[ID])-1
        if end_idx > end_sig_idx:
            right_pad = np.zeros((end_idx - end_sig_idx,self.n_channels))
            end_idx = end_sig_idx
        x = np.zeros(self.dim)
        
        if usecopy:
            sig = self.copy[ID][start_idx:end_idx,:self.n_channels]
        else:
            sig = self.data[ID][start_idx:end_idx,:self.n_channels]

        if left_pad is None and right_pad is None:
            x = sig
        elif left_pad is not None and right_pad is None:
            x = np.concatenate((left_pad,sig))
        elif left_pad is None and right_pad is not None:
            x = np.concatenate((sig,right_pad))

        # Data quality checks
        assert not np.any(np.isnan(x)), "Data contains a NaN value!"
        assert not np.any(np.isinf(x)), "Data contains an infinity value!"
        y = np.eye(self.n_classes)[int(label)]
        assert x.shape == self.dim, f"x shape {x.shape} does not match expected shape {self.dim}"
        return x,y



if __name__ == "__main__":
    dg = DataGeneratorApneaTest(data_path = '/Users/danielyaeger/Documents/raw_no_baseline',
                                  mode = 'train',
                                  single_ID = 'XAXVDJYND7Q6JTK',
                                  k_fold = 3, 
                                  test_index = 2,
                                  sampling_rate=10)
    
    
    X, y = dg.get_data()

