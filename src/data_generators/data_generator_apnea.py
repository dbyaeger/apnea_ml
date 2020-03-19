#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:28:42 2019

@author: danielyaeger
"""
from pathlib import Path
import numpy as np
import pickle
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGeneratorApnea(Sequence):
    'Data Generator for TensorFlow'
    def __init__(self, data_path: str = '/Users/danielyaeger/Documents/raw_apnea_data',
                 batch_size: int = 64, mode: str = 'train', sampling_rate: int = 10, 
                 n_classes = 2, desired_number_of_samples = 2.1e6,
                 use_staging = True,
                 context_samples: int = 300, shuffle: bool = False, 
                 load_all_data: bool = True, single_ID = None, REM_only: bool = False):

        assert mode in ['train', 'cv', 'test', 'val'], f'mode must be train, cv, val, or test, not {mode}'

        if type(data_path) == str: data_path = Path(data_path)
        self.data_path = data_path
        assert self.data_path.is_dir(), f"{data_path} is not a valid path"
        
        with self.data_path.joinpath('channel_list.p').open('rb') as fcl:
            channel_list = pickle.load(fcl)
            self.n_channels = len(channel_list)
        
        with self.data_path.joinpath('stage_dict.p').open('rb') as fs:
            self.stage_dict = pickle.load(fs)
        
        with self.data_path.joinpath('master_ID_list.p').open('rb') as fd:
            partition = pickle.load(fd)
            
        with self.data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
            targets = pickle.load(ta)
        
        self.targets = targets
        
        self.total_samples = sum([len(self.targets[ID]) for ID in self.targets])
            
        if mode == "train":
            self.IDs = partition["train"]
        elif mode in ["cv", "val"]:
            self.IDs = partition["val"] if "val" in partition else partition["cv"]
        elif mode == "test":
            self.IDs = partition["test"]
        
        self.single_ID = single_ID
        if single_ID is not None:
            assert type(single_ID) == str, 'Single ID {single_ID} must be a string!'
            self.IDs = [single_ID]
            
            new_target = self.targets[single_ID]
            del self.targets
            self.targets = {self.single_ID: new_target}
            
            new_stages = self.stage_dict[single_ID]
            del self.stage_dict
            self.stage_dict = {self.single_ID: new_stages}
        
        assert n_classes in [2,3], 'n_classes must be 2 or 3!'
        self.n_classes = n_classes
        
        if self.n_classes == 3:
            self.label_dict = {'0': 0, 'H': 1, 'A': 2}
       
        elif self.n_classes == 2:
            self.label_dict = {'0': 0, 'A/H': 1}
            
            for ID in self.targets:
                idx = np.nonzero(self.targets[ID] > 1)[0]
                if len(idx) > 0:
                    self.targets[ID][idx] = 1    
        
        # Exclude Wake and None from targets
        if use_staging:
            for ID in self.stage_dict:
                for epoch in self.stage_dict[ID]:
                    if not REM_only:
                        if self.stage_dict[ID][epoch] in [None, 'W']:
                            self.targets[ID][(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1
                    elif REM_only:
                        if self.stage_dict[ID][epoch] != 'R':
                            self.targets[ID][(epoch-1)*sampling_rate*30:epoch*sampling_rate*30] = -1

        self.inverse_label_dict = {self.label_dict[label]: label for label in self.label_dict}
        
        self.load_all_data = load_all_data
        if load_all_data:
            self.data = {}
            for ID in self.IDs:
                self.data[ID] = np.load(str(self.data_path.joinpath(ID + '.npy')))
                
        
        self.context_samples = context_samples
        self.batch_size = batch_size
        self.fs = sampling_rate
        self.shuffle = shuffle
                
        self.dim = (2*context_samples + 1, self.n_channels)
        self.mode = mode
        
         # Create samples
        self.desired_number_of_samples = desired_number_of_samples
        self.make_samples()
                
        # Oversample through data augmentation
        self.on_epoch_end()
     
    def make_samples(self):
        """If mode is set to train, creates samples by randoming sampling each
        label in proportion to the total amount of samples for each sleeper"""
        # Randomly sample minority classes to bring count up to level of majority class
        self.samples = []
        if self.mode == 'train' and self.single_ID is None:
            # ratio of training samples per sample
            ratio = self.desired_number_of_samples/self.total_samples
            for ID in self.IDs:
                num_samples = int(ratio*len(self.targets[ID]))
                
                # Add zero samples
                indexes = np.nonzero(self.targets[ID] == 0)[0]
                np.random.shuffle(indexes)
                if len(np.nonzero(self.targets[ID] > 0)[0]) > 0:
                    samp_list = [(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:(num_samples//2)]]
                else:
                    samp_list = [(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:num_samples]]
                      
                # Add A/H samples
                indexes = np.nonzero(self.targets[ID] > 0)[0]
                np.random.shuffle(indexes)
                samp_list.extend([(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:(num_samples//2)]])

                # Add samples to master list
                self.samples.extend(samp_list)
        elif self.mode == 'train' and self.single_ID is not None:
            
            ID = self.single_ID
            
            # Add A/H samples
            indexes = np.nonzero(self.targets[ID] > 0)[0]
            samp_list = [(ID, idx, int(self.targets[ID][idx])) for idx in indexes]
            
            # Add equal number of zero samples
            indexes = np.nonzero(self.targets[ID] == 0)[0]
            np.random.shuffle(indexes)
            samp_list.extend([(ID, idx, int(self.targets[ID][idx])) for idx in indexes[0:len(samp_list)]])
            
            # Add samples to master list
            self.samples.extend(samp_list)

        else:
            for ID in self.IDs:
                for idx in range(self.targets[ID].shape[0]):
                    if self.targets[ID][idx] >= 0:
                        self.samples.append((ID, idx, int(self.targets[ID][idx])))
  
    @property
    def labels(self):
        "Returns labels for samples. Safeest to use when shuffle set to False"
        return [x[-1] for x in self.samples]
    
    @property
    def all_targets(self):
        "Returns targets even for where there is no prediction"
        return self.targets
    
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
            class_weights[label] = max(list(class_counts.values()))/class_counts[label]
        return class_weights
    
    @property    
    def prior_prob(self):
        "Calculates prior probabilities and counts of each label"
        # Generate prior probabilities and weights
        label_counts = {label:0 for label in self.inverse_label_dict}
        for ID in self.targets:
            for label in self.inverse_label_dict:
               label_counts[label] += np.nonzero(self.targets[ID] == label)[0].shape[0]
        prior_prob = {label:(label_counts[label]/self.total_samples) for label in label_counts}
        return prior_prob
    
    def on_epoch_end(self):  
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)
  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.indexes)//self.batch_size
    
    def __getitem__(self, index):
        """Generate one batch of data deterministically or randomly"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.samples[k] for k in indexes]

        # Make list_IDs_temp an instance variable for error checking
        self.list_IDs_temp = list_IDs_temp         
        
        return self.__data_generation(list_IDs_temp)
            
    
    def __data_generation(self,list_IDs_temp):
        """Generate one batch of samples"""
        X = np.zeros((self.batch_size, *self.dim), dtype=np.float64)
        y = np.zeros((self.batch_size, self.n_classes))
        for i, item in enumerate(list_IDs_temp):
            ID, epoch, label = item
            features, label = self._sample(ID, epoch, label)
            X[i], y[i] = features, label

        # self._check_labels(y)
        return X, y
    
    def _sample(self, ID, center_idx, label):
        """ Returns random or weighted random sampled from ID
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
        
        if not self.load_all_data:
            sig = np.load(file=str(self.data_path.joinpath(ID + '.npy')),mmap_mode='r')[start_idx:end_idx,:self.n_channels]
        elif self.load_all_data:
            sig = self.data[ID][start_idx:end_idx,:self.n_channels]

        if left_pad is None and right_pad is None:
            x = sig
        elif left_pad is not None and right_pad is None:
            x = np.concatenate((left_pad,sig))
        elif left_pad is None and right_pad is not None:
            x = np.concatenate((sig,right_pad))

        # Data quality checks
        #assert not np.any(np.isnan(x)), "Data contains a NaN value!"
        #assert not np.any(np.isinf(x)), "Data contains an infinity value!"
        assert label >= 0, f'{label} less than zero!'
        y = np.eye(self.n_classes)[int(label)]
        #assert x.shape == self.dim, f"x shape {x.shape} does not match expected shape {self.dim}"
        return x,y

class DataGeneratorApneaRandomSubset(DataGeneratorApnea):
    def __init__(self, percentage_to_sample: float = 0.2, **args):
        super().__init__(**args)
        self.percentage = percentage_to_sample
    
    def on_epoch_end(self):  
        self.indexes = np.arange(len(self.samples))
        np.random.shuffle(self.indexes)
        
        # Get indices for positive and negative samples
        apnea_indices = [x for x in self.indexes if self.samples[-1] > 0]
        none_indices = [x for x in self.indexes if self.samples[-1] == 0]
        
        # Take same percentage of each
        apnea_indices = apnea_indices[:int(len(apnea_indices)*self.percentage)]
        none_indices = none_indices[:int(len(none_indices)*self.percentage)]
        
        # Combine arrays and convert to numpy array
        self.indexes = np.array(apnea_indices + none_indices)

def check_data_generator(data_generator, index=0):
    "Checks the output of the data generator"
    targets = dg.targets
    x, y = dg.__getitem__(index)
    orders = dg.list_IDs_temp
    for i,order in enumerate(orders):
        ID, idx,label = order
        print(f'order:{order}\ty:{int(y[i].argmax())}\ty_target:{int(targets[ID][idx])}')
        assert idx <= len(targets[ID]), f"idx {idx} from outside of target[{ID}] of length {len(targets[ID])}"
        assert targets[ID][idx] == y[i].argmax(), f"target is supposed to be {targets[ID][idx]} but it is {y[i].argmax()}"
        assert len(y) == len(orders) == dg.batch_size, f"Batchsize is {dg.batch_size}\tlength of y: {len(y)}\tlength of orders: {len(orders)}"
        #print(x)
        # X = np.load(dg.data_path.joinpath(ID+'.npy'))
        # X = X[idx - dg.context_samples:idx + dg.context_samples,:14]
        # assert np.allclose(X,x[i,:,:]), f"idx: {idx}\nX: {X}\nx[{i}]: {x[i,:,:]}"
    

if __name__ == "__main__":
    dg = DataGeneratorApnea(data_path = '/Users/danielyaeger/Documents/raw_apnea_data',
                                  mode = 'train',
                                  batch_size=128,
                                  shuffle = True,
                                  sampling_rate=10,
                                  load_all_data = True,
                                  desired_number_of_samples = 2.1e6)
    
    print(f'class weights: {dg.class_weights}')
    print(f'Number of batches per epoch: {len(dg)}')
    num_rounds = len(dg)
    round_dict = {}
    for i in range(num_rounds):
        print(i)
        check_data_generator(dg,i)



