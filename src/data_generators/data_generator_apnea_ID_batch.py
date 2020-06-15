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

class DataGeneratorApneaIDBatch(Sequence):
    'Data Generator for TensorFlow'
    def __init__(self, data_path: str = '/Users/danielyaeger/Documents/raw_apnea_data',
                 batch_size: int = 64, mode: str = 'train', sampling_rate: int = 10, 
                 n_classes = 2, desired_number_of_samples = 2.1e6, load_all_data: bool = True,
                 use_staging: bool = True, select_channels: list = 'all',
                 context_samples: int = 300, shuffle: bool = False, 
                 single_ID = None, REM_only: bool = False, normalizer: callable = None):

        assert mode in ['train', 'cv', 'test', 'val'], f'mode must be train, cv, val, or test, not {mode}'

        if type(data_path) == str: data_path = Path(data_path)
        self.data_path = data_path
        assert self.data_path.is_dir(), f"{data_path} is not a valid path"
        
        with self.data_path.joinpath('channel_list.p').open('rb') as fcl:
            channel_list = pickle.load(fcl)
            if select_channels == 'all':
                self.n_channels = len(channel_list)
            else:
                self.n_channels = len(select_channels)
        
        with self.data_path.joinpath('stage_dict.p').open('rb') as fs:
            self.stage_dict = pickle.load(fs)
        
        with self.data_path.joinpath('master_ID_list.p').open('rb') as fd:
            partition = pickle.load(fd)
            
        with self.data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as ta:
            targets = pickle.load(ta)
        
        self.targets = targets
        self.normalizer = normalizer
        
        self.total_samples = sum([len(self.targets[ID]) for ID in self.targets])
            
        if mode == "train":
            self.IDs = partition["train"]
        elif mode in ["cv", "val"]:
            self.IDs = partition["val"] if "val" in partition else partition["cv"]
        elif mode == "test":
            self.IDs = partition["test"]
        
        self.IDs = list(self.IDs)
        
        self.single_ID = single_ID
        if single_ID is not None:
            assert type(single_ID) == str, 'Single ID {single_ID} must be a string!'
            # Generate an error if single_ID is not in the given partition
            assert single_ID in self.IDs, f'{single_ID} not in partition {mode}!'
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
                        if self.stage_dict[ID][epoch] not in ['R','1','2','3']:
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
        if isinstance(select_channels,list):
            self.channel_idx = []
            self.select_channels = select_channels
            for channel in select_channels:
                self.channel_idx.append(channel_list.index(channel))
        elif select_channels == 'all':
            self.channel_idx = np.arange(len(channel_list))
            
                
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
    def get_all_data(self):
        """Returns all data in samples"""
        all_data = np.zeros((len(self.samples),self.n_channels))
        print(f'index: {self.channel_idx}')
        count = 0
        for ID in self.IDs:
            sample_idx = [idx for (ID, idx, _) in self.samples if ID == ID]
            print(f'sample_idx length: {len(sample_idx)}')
            data = np.load(str(self.data_path.joinpath(ID + '.npy')))
            print(f'Shape of spot: {all_data[count:count+len(sample_idx),:].shape}')
            print(f'Shape of data: {data[sample_idx,self.channel_idx].shape}')
            all_data[count:count+len(sample_idx),:] = data[sample_idx,self.channel_idx]
            count += len(sample_idx)
        return all_data
            
    @property
    def labels(self):
        """Returns labels for samples. Safeest to use when shuffle set to False"""
        return [x[-1] for x in self.samples]
    
    @property
    def all_targets(self):
        """Returns targets even for where there is no prediction"""
        return self.targets
    
    @property
    def class_weights(self):
        """Calculates class_weights assuming apnea/hypopnea and None as classes.
        As sampling is per ID, weights for apnea/hypopnea are calculated as 
        inverse of average imbalance ratio
        
                    (# apnea/hypopnea samples/# none samples)
        across IDs
        """
        apnea_ratio = 0
        # Get fraction by ID:
        for ID in self.IDs:
            ID_samples = list(filter(lambda x: x[0] == ID, self.samples))
            apnea_sample_num = len(list(filter(lambda x: x[-1] > 0, ID_samples)))
            none_sample_num = len(list(filter(lambda x: x[-1] == 0, ID_samples)))
            
            # calculate ratio of apneic to non-apneic events
            apnea_ratio += apnea_sample_num/none_sample_num
        
        # Normalize to number of IDs
        apnea_ratio = apnea_ratio/len(self.IDs)
        
        class_weights = {0: 1, 1: (1/apnea_ratio)}

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
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.IDs)
    
    def __getitem__(self, index):
        """Generate one batch of data deterministically or randomly"""
        ID_idx = self.indexes[index]
        return self.__data_generation(self.IDs[ID_idx])
            
    
    def __data_generation(self,ID):
        """Generate one batch of samples"""
        # Load data
        self.data = np.load(str(self.data_path.joinpath(ID + '.npy')))
        
        # Get indices of samples and randomly sample
        ID_samples = list(filter(lambda x: x[0] == ID, self.samples))
        ID_samples = ID_samples[:self.batch_size]
    
        X = np.zeros((self.batch_size, *self.dim), dtype=np.float64)
        y = np.zeros((self.batch_size, self.n_classes))
        for i, item in enumerate(ID_samples):
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
        
        sig = self.data[start_idx:end_idx,self.channel_idx]
        
        if self.normalizer is not None:
            sig = self.normalizer.transform(sig)

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

class DataGeneratorApneaAllWindows(DataGeneratorApneaIDBatch):
    """Returns all of the data for each ID."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        """Generate one batch of data deterministically or randomly"""
        ID_idx = self.IDs[index]
        return self.__data_generation(self.IDs[ID_idx])
            
    
    def __data_generation(self,ID):
        """Generate one batch of samples"""
        # Load data
        self.data = np.load(str(self.data_path.joinpath(ID + '.npy')))
        
        # Get indices of samples and randomly sample
        ID_samples = list(filter(lambda x: x[0] == ID, self.samples))
        
        # Ensure samples are sorted by index
        ID_samples.sort(key = lambda x: x[1])
    
        X = np.zeros((len(ID_samples), *self.dim), dtype=np.float64)
        y = np.zeros((len(ID_samples), self.n_classes))
        for i, item in enumerate(ID_samples):
            ID, epoch, label = item
            features, label = self._sample(ID, epoch, label)
            X[i], y[i] = features, label

        return X, y
    
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
    dg = DataGeneratorApneaIDBatch(data_path = '/Users/danielyaeger/Documents/raw_apnea_data',
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



