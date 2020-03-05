#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:33:31 2019

@author: danielyaeger

Collection of functions for getting statistics on stages and checking whether
p files were made correctly. Also contains files to check on the 
"""

import openxdf
import numpy as np
import pandas as pd
from pathlib import Path
from biosppy.signals.ecg import christov_segmenter, ecg, gamboa_segmenter, hamilton_segmenter, engzee_segmenter
import pickle
from matplotlib import pyplot as plt
from utils.open_xdf_helpers import load_data, select_epoch
import more_itertools
from scipy.signal import butter, lfilter, filtfilt, firwin, iirnotch
from utils.dsp import design_filter, filter_and_resample, lowpass
from resampy import resample
from analysis import Analyzer2
from utils.baseliner import baseline
from artifactreduce.artifact_reduce import ArtifactReducer

def lowpass(x, cutoff, fs):
    B = firwin(65, cutoff, fs=fs)
    x = filtfilt(B, [1.], x)
    return x

def notch(x, f0 = 60, Q = 30, fs=200):
    w0 = f0/(fs/2)
    b, a = iirnotch(w0=w0, Q=Q)
    x = filtfilt(b, a, x)
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def examine_ecg(ID = None, algorithm = ecg,
                ID_list_path = '/Users/danielyaeger/Documents/filtered_npy_apnea/ID_partitions.p',
                path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data'):
    """Opens either a random .nkamp file and analyzes ecg r-peaks using the
    given algorithm, or randomly selects an ID.
    """
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    with open(ID_list_path, 'rb') as fh: ID_dict = pickle.load(fh)
    
    if ID is None:
        ID_list = []
        for partition in ID_dict:
            ID_list.extend(ID_dict[partition])
        
        ID = np.random.choice(ID_list)
    print(f'Plotting ECG signals for ID: {ID}')
    
     # Make paths
    path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
    
    # Get ECG signal from nkamp file
    print('Load .xdf and .nkamp signals')
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    ecg_signal = signal.read_file('ECG')
    ecg_signal = ecg_signal['ECG'][0:1000].ravel()
    
    # Run algorithm
    data = algorithm(signal = ecg_signal, sampling_rate = 200, show = True)
    
    return data
    
    
    
    
    
    

def find_negative_processed_regions(path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                 channels = ['SpO2'],
                 ID_list_path = '/Users/danielyaeger/Documents/filtered_npy_apnea/ID_partitions.p',
                 resampled = True):
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    with open(ID_list_path, 'rb') as fh: ID_dict = pickle.load(fh)
    
    ID_list = []
    for partition in ID_dict:
        ID_list.extend(ID_dict[partition])
    
    ID = np.random.choice(ID_list)
    print(f'ID: {ID}')
    
    # Make paths
    path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
    
    # Get signal from nkamp file
    print('Load .xdf and .nkamp signals')
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    signals = signal.read_file(channels)
    print(f'Shape of signals: {signals[channels[0]].shape}')
    for channel in signals:
        signals[channel] = signals[channel]
 
            
    print('About to filter')
    new_channels = []   
    for channel in channels:
        new_channels.append(channel)
        new_channels.append(f'{channel}_processed')
        if channel == 'SpO2':
            sr_in, sr_out = signals[channel].shape[-1], 10
            signals[channel] = signals[channel].ravel()
            sig = resample(signals[channel], sr_in, sr_out, axis=-1)
            sig = sig*0.3663*0.001
            signals[channel] = signals[channel]
            signals[f'{channel}_processed'] = baseline(data=sig,sampling_rate=10,quantile=0.95,baseline_length=120,step_size=10)
            
    # Find negative regions and plot
    idx = np.nonzero(signals['SpO2_processed'] == 0)[0]
    
    if len(idx) == 0:
        print('No negative regions found')
        print('Calling function again')
        find_negative_processed_regions()
    
    idx_0 = (idx[idx > 1000])[0]
    
    
    middle = idx_0//10
    start_time =  middle - 100
    end_time = middle + 100
    
    signals['SpO2'] = signals['SpO2'][start_time*25:end_time*25]
    print(f"Shape of signals[SpO2]: {signals['SpO2'].shape}")
    signals['SpO2_processed'] = signals['SpO2_processed'][start_time*10:end_time*10]
    print(f"Shape of signals[SpO2_processed]: {signals['SpO2_processed'].shape}")
    
    
    print(new_channels)
    for i,channel in enumerate(new_channels):
        plt.subplot(len(new_channels),1,i+1)
        plt.plot(signals[channel],label=f'{channel}')
        if channel == 'SpO2_processed':
            plt.plot([0]*len(signals[channel]), label='Zero-line')
        plt.legend(loc='best')

def check_processing(path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                 channels = ['Abdomen'],
                 ID_list_path = '/Users/danielyaeger/Documents/filtered_npy_apnea/ID_partitions.p',
                 resampled = True):
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    with open(ID_list_path, 'rb') as fh: ID_dict = pickle.load(fh)
    
    ID_list = []
    for partition in ID_dict:
        ID_list.extend(ID_dict[partition])
    
    ID = np.random.choice(ID_list)
    print(f'ID: {ID}')
    
    # Make paths
    path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
    
    # Get signal from nkamp file
    print('Load .xdf and .nkamp signals')
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    signals = signal.read_file(channels)
    length = len(signals[channels[0]])
    print(f'Shape of signals: {signals[channels[0]].shape}')
    start = np.random.choice(np.arange(length-300))
    end = start + 300
    print(f'Start: {start}\tEnd: {end}')
    for channel in signals:
        signals[channel] = signals[channel]
 
    analyzer = Analyzer2(100)
    print('About to filter')
    new_channels = []   
    for channel in channels:
        new_channels.append(channel)
        if channel in ['Airflow','P-Flo','Abdomen','Chest']:
            new_channels.append(f'{channel}_rms_energy')
            new_channels.append(f'{channel}_max')
            sig = signals[channel].ravel()
            sig = lowpass(signals[channel],3,200)
            B, A = design_filter(200, 50)
            sig = filter_and_resample(sig, B, A)
            signals[channel] = sig
            t,features = analyzer.analyze(sig, emg_filtering = False,
                                                                  nsize = 5000, nrate = 10)
            tx = np.arange(len(sig)//10)
            features = np.array([np.interp(tx, t, features) for features in features.T]).T 
            #features = baseline(data=features,sampling_rate=10,quantile=0.5,baseline_length=120,step_size=10)
            signals[f'{channel}_rms_energy'] = features[10*start:10*end,1]
            signals[f'{channel}_max'] = features[10*start:10*end,0]
            print(len(signals[f'{channel}_max']))
            signals[f'abs_val({channel})']= np.abs(signals[channel][100*start:100*end])
            del signals[channel]
            new_channels.pop(0)
            new_channels.append(f'abs_val({channel})')
           
        if channel == 'SpO2':
            new_channels.append(f'{channel}_processed')
            sr_in, sr_out = signals[channel].shape[-1], 10
            signals[channel] = signals[channel].ravel()
            sig = resample(signals[channel], sr_in, sr_out, axis=-1)
            sig = sig*0.3663*0.001
            signals[channel] = signals[channel][sr_in*start:sr_in*end]
            signals[f'{channel}_processed'] = baseline(data=sig,sampling_rate=10,quantile=0.95,baseline_length=120,step_size=10)[sr_out*start:sr_out*end]
            
    print(new_channels)
    for i,channel in enumerate(new_channels):
        plt.subplot(len(new_channels),1,i+1)
        if len(signals[channel]) == (end-start)*10:
            fs = 10
        else:
            fs = 100
        plt.plot(np.arange(len(signals[channel]))*(1/fs),signals[channel],label=f'{channel}')
        plt.legend(loc='best')

def check_filtering(path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                 channels = ['Abdomen','Chest','Airflow', 'P-Flo'],
                 ID_list_path = '/Users/danielyaeger/Documents/filtered_npy_apnea/ID_partitions.p',
                 artifact_reduce = True,
                 resampled = True,
                 low_pass = True):
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    with open(ID_list_path, 'rb') as fh: ID_dict = pickle.load(fh)
    
    ID_list = []
    for partition in ID_dict:
        ID_list.extend(ID_dict[partition])
    
    ID = np.random.choice(ID_list)
    
    # Make paths
    path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
    
    # Get signal from nkamp file
    print('Load .xdf and .nkamp signals')
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    
    temp_channels = channels.copy()
    temp_channels.append('ECG')
    
    signals = signal.read_file(temp_channels)
    length = len(signals[channels[0]])
    print(f'Length of signals: {length}')
    start = np.random.choice(np.arange(length-30))
    end = start + 30
    print(f'Start: {start}\tEnd: {end}')
    for channel in signals:
        if channel != 'SpO2':
            signals[channel] = signals[channel].ravel()[start*200:end*200]
            print(f'{channel} dimension: {signals[channel].shape}')
        else:
            signals[channel].ravel()[start*25:end*25]
            print(f'{channel} dimension: {signals[channel].shape}')
            
    # Do filtering for ecg signal
    signals['ECG'] = resample(signals['ECG'], 200, 100)
    
            
    print('About to filter')
    new_channels = []   
    for channel in channels:
        new_channels.append(channel)
        new_channels.append(f'{channel}_filtered')
        if channel in ['Airflow','P-Flo','Abdomen','Chest']:
            signals[f'{channel}_filtered'] = signals[channel]
            if low_pass:
                signals[f'{channel}_filtered'] = lowpass(signals[channel],3,200)
            if resampled:
                signals[f'{channel}_filtered'] = resample(signals[f'{channel}_filtered'], 200, 100)
            if artifact_reduce:
                ar = ArtifactReducer()
                ar.fit(ecg = signals['ECG'], emg = signals[f'{channel}_filtered'])
                [emg_filt, noise_reduction] = ar.process_arrays(ecg = signals['ECG'], emg = signals[f'{channel}_filtered'])
                signals[f'{channel}_filtered'] = emg_filt

    for i,channel in enumerate(new_channels):
        plt.subplot(len(new_channels),1,i+1)
        plt.plot(signals[channel],label=f'{channel}')
        plt.legend(loc='best')
    plt.show()

def load_xdf(xdf_path):
    xdf = openxdf.OpenXDF(xdf_path)
    return xdf

def get_stages(xdf, scorer="Dennis"):
    staging = xdf.dataframe(epochs=True, events=False)
    staging = staging[staging["Scorer"] == scorer]
    epoch_number = np.array(staging['EpochNumber'])
    stages = np.array(staging['Stage'])
    return epoch_number, stages

def check_lengths(path_to_p_files = '/Volumes/Elements/sleep_staging'):
    """
    Goes through .p files and reports files for which the length of the singal
    array is not equal to the final epoch * epoch length
    """
    EPOCH_LEN = 30
    
    if not isinstance(path_to_p_files, Path): path_to_p_files = Path(path_to_p_files)
    
    p_files = [f for f in path_to_p_files.iterdir() if 'X' in f.stem]
     
    for p_file in p_files:
        with p_file.open('rb') as fh: data = pickle.load(fh)
        
        print(f"Checking ID:{p_file.name.split('.')[0]}")
        max_epoch = max(list(data['stages'].keys()))
        
        if (max_epoch*EPOCH_LEN) != data['signals']['Chin'].shape[0]:
            print(f"\tMismatch! Last epoch:{max_epoch}\tLast epoch X 30:{max_epoch*EPOCH_LEN}\tLength of Chin signal:{data['signals']['Chin'].shape[0]}")

def check_data_generator_numpy(path_to_npy_files = '/Volumes/TOSHIBA EXT/selected_p_files_npy',
                         path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                         channel = 'EOG-L',
                         context_epochs = 5,
                         batch_size = 128,
                         augment = True,
                         plot = False,
                         identity_test = True):
    """Checks whether the files returned by data generator correspond to the 
    original nkamp files by looking at four files at a time and one channel
    at a time.
    """
    EPOCH_LEN = 30
    fs = 100
    
    if plot:
        assert batch_size == 4, "Batch size must be 4 if plotting"
    
    channel_index = sorted(['O2-A1','EOG-L', 'C4-A1', 'F4-A1', 'C3-A2', 'F3-A2', 'O1-A2', 'EOG-R']).index(channel)
    
    if not isinstance(path_to_npy_files, Path): path_to_npy_files = Path(path_to_npy_files)
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
     # Instantiate data generator
    dg = DataGeneratorStagesNumPy(data_path = path_to_npy_files, batch_size = batch_size, context_epochs = context_epochs, augment = augment)
    X,Y = dg.__getitem__(0)
    
    # Take data in X corresponding to signal
    print('fetching data from data generator')
    X = X[:,:,channel_index]
    dg_fs = X.shape[1]//30
    
    # Get names of files
    file_data = dg.list_IDs_temp
    
    # Instantiate plot
    if plot:
        plt.figure()
        counter = 1
    
    for i,datum in enumerate(file_data):
        ID, epoch, stage = datum
            
        # Load xdf and nkamp files
        path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
        path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
        xdf, signal = load_data(path_to_xdf, path_to_signal)
    
        # Get stage
        _ , stages = get_stages(xdf)
        assert stage == stages[int(epoch)-1], f"Stage in p file: {stage} does not match stage in xdf file: {stages[epoch-1]}!"
        assert stage == dg.inverse_label_dict[np.argmax(Y[i,:])], f"Stage in p file: {stage} does not match stage from data generator: {dg.inverse_label_dict(np.argmax(Y[i,:]))}"
        
        # Get signal from nkamp file
        print('fetching data from .nkamp files')
        nsig_data = signal.read_file([channel])
        nsig_data = np.concatenate(nsig_data[channel])
        
        ## Epochs need not be whole numbers with data augmentation method
        center_idx = np.round((epoch - 0.5)*EPOCH_LEN*fs,0)
       
        # Initialize padding variable
        left_pad, right_pad = None, None
        
        # Zero-pad on left if necessary
        start_idx = int(center_idx - (context_epochs+0.5)*EPOCH_LEN*fs)
        if start_idx < 0:
            left_pad = np.zeros(-start_idx)
            start_idx = 0
            
        # Zero-pad on right if necessary
        end_idx = int(center_idx + (context_epochs+0.5)*EPOCH_LEN*fs)
        end_sig_idx = nsig_data.shape[0]
        if end_idx > end_sig_idx:
            right_pad = np.zeros(end_idx - end_sig_idx)
            end_idx = end_sig_idx
                            
        nsig_data = nsig_data[start_idx:end_idx]
        if left_pad is not None and right_pad is None:
            nsig_data = np.concatenate((left_pad,nsig_data))
        elif left_pad is None and right_pad is not None:
            nsig_data = np.concatenate((nsig_data,right_pad))
        
        nsig_fs = len(nsig_data)//30
        
        if identity_test:
            print('testing for equality')
            assert np.allclose(X[i,:],nsig_data,), ".npy data and .nkamp data not the same!"
        
        # Plot data
        if plot:
            plt.subplot(4,2,counter)
            counter += 1
            plt.plot(np.arange(nsig_fs*30)/nsig_fs, nsig_data, label = f'ID: {ID} .nkamp')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.subplot(4,2,counter)
            counter += 1
            plt.plot(np.arange(dg_fs*30)/dg_fs, X[i,:], label = f'ID: {ID} .p')
            plt.legend(loc='best')
            plt.tight_layout()
    if plot:
        plt.show()

def plot_signals(path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                 channels = ['Abdomen', 'Airflow', 'Chest', 'ECG', 'P-Flo', 'Snore', 'SpO2'],
                 ID_list_path = '/Users/danielyaeger/Documents/filtered_npy_apnea/ID_partitions.p'):
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    with open(ID_list_path, 'rb') as fh: ID_dict = pickle.load(fh)
    
    ID_list = []
    for partition in ID_dict:
        ID_list.extend(ID_dict[partition])
    
    ID = np.random.choice(ID_list)
    
    # Make paths
    path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
    
    # Get signal from nkamp file
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    signals = signal.read_file(channels)
    for channel in signals:
        print(f'{channel}\t shape: {signals[channel].shape}')
    
    length = len(signals[channels[0]])
    
    start = np.random.choice(np.arange(length-30))
    end = start + 30
    
    for i,channel in enumerate(channels):
        plt.subplot(len(channels),1,i+1)
        if channel != 'SpO2':
            plt.plot(signals[channel].ravel()[start*200:end*200],label=f'{channel}')
        else:
            plt.plot(signals[channel].ravel()[start*25:end*25],label=f'{channel}')
        plt.legend(loc='best')
    
        
    
    
    
    
    
def check_data_generator(path_to_p_files = '/Volumes/Elements/sleep_staging',
                         path_to_nkamp_files = '/Volumes/TOSHIBA EXT/training_data',
                         channel = 'C3-A2',
                         context_epochs = 5,
                         augment = True):
    """Checks whether the files returned by data generator correspond to the 
    original nkamp files by looking at four files at a time and one channel
    at a time.
    """
    EPOCH_LEN = 30
    fs = 100
    
    channel_index = sorted(['O2-A1','EOG-L', 'C4-A1', 'F4-A1', 'C3-A2', 'F3-A2', 'O1-A2', 'EOG-R']).index(channel)
    
    if not isinstance(path_to_p_files, Path): path_to_p_files = Path(path_to_p_files)
    
    if not isinstance(path_to_nkamp_files, Path): path_to_nkamp_files = Path(path_to_nkamp_files)
    
    xdf, signal = load_data(path_to_xdf, path_to_signal)
     
     
    
    # Instantiate data generator
    dg = DataGeneratorStages(batch_size = 4, context_epochs = context_epochs, augment = augment)
    X,Y = dg.__getitem__(0)
    
    # Take data in X corresponding to signal
    X = X[:,:,channel_index]
    dg_fs = X.shape[1]//30
    
    # Get names of files
    file_data = dg.list_IDs_temp
    
    # Instantiate plot
    plt.figure()
    counter = 1
    
    for i,datum in enumerate(file_data):
        ID, epoch, stage = datum
            
        # Load xdf and nkamp files
        path_to_xdf = path_to_nkamp_files.joinpath(ID + '.xdf')
        path_to_signal = path_to_nkamp_files.joinpath(ID + '.nkamp')
        xdf, signal = load_data(path_to_xdf, path_to_signal)
    
        # Get stage
        _ , stages = get_stages(xdf)
        assert stage == stages[int(epoch)-1], f"Stage in p file: {stage} does not match stage in xdf file: {stages[epoch-1]}!"
        assert stage == dg.inverse_label_dict[np.argmax(Y[i,:])], f"Stage in p file: {stage} does not match stage from data generator: {dg.inverse_label_dict(np.argmax(Y[i,:]))}"
        
        # Get signal from nkamp file
        nsig_data = signal.read_file([channel])
        nsig_data = np.concatenate(nsig_data[channel])
        
        ## Epochs need not be whole numbers with data augmentation method
        center_idx = np.round((epoch - 0.5)*EPOCH_LEN*fs,0)
       
        # Initialize padding variable
        left_pad, right_pad = None, None
        
        # Zero-pad on left if necessary
        start_idx = int(center_idx - (context_epochs+0.5)*EPOCH_LEN*fs)
        if start_idx < 0:
            left_pad = np.zeros(-start_idx)
            start_idx = 0
            
        # Zero-pad on right if necessary
        end_idx = int(center_idx + (context_epochs+0.5)*EPOCH_LEN*fs)
        end_sig_idx = nsig_data.shape[0]
        if end_idx > end_sig_idx:
            right_pad = np.zeros(end_idx - end_sig_idx)
            end_idx = end_sig_idx
                            
        nsig_data = nsig_data[start_idx:end_idx]
        if left_pad is not None and right_pad is None:
            nsig_data = np.concatenate((left_pad,nsig_data))
        elif left_pad is None and right_pad is not None:
            nsig_data = np.concatenate((nsig_data,right_pad))
        
        nsig_data = nsig_data()
        nsig_fs = len(nsig_data)//30
        
        # Plot data
        plt.subplot(4,2,counter)
        counter += 1
        plt.plot(np.arange(nsig_fs*30)/nsig_fs, nsig_data, label = f'ID: {ID} .nkamp')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.subplot(4,2,counter)
        counter += 1
        plt.plot(np.arange(dg_fs*30)/dg_fs, X[i,:], label = f'ID: {ID} .p')
        plt.legend(loc='best')
        plt.tight_layout()
    
    plt.show()

def check_data_partition(path_to_p_files = '/Volumes/Elements/sleep_staging'):
    """ Checks that data_partition.p contains the same staging and epoch information
    as data['staging'] in each .p file.
    """
    if not isinstance(path_to_p_files, Path): path_to_p_files = Path(path_to_p_files)
    
    with path_to_p_files.joinpath('data_partition2.p').open('rb') as fh: partitions = pickle.load(fh)
    
    for partition in partitions:
        IDs = set([x[0] for x in partitions[partition]])
        for ID in IDs:
            observations = [x for x in partitions[partition] if x[0] == ID]
            with path_to_p_files.joinpath(ID+'.p').open('rb') as fp: data = pickle.load(fp)
            for obs in observations:
                assert obs[1] in data['stages'], f"Epoch {obs[1]} not found in {ID}.p!"
                assert data['stages'][obs[1]] == obs[2], f"For epoch {obs[1]}, stage in .p file is {data['stages'][obs[1]]} but stage in data_partition.p is {obs[2]}!"


def correct_data_partition(path_to_p_files = '/Volumes/Elements/sleep_staging'):
    """ Checks that data_partition.p contains the same staging and epoch information
    as data['staging'] in each .p file.
    """
    if not isinstance(path_to_p_files, Path): path_to_p_files = Path(path_to_p_files)
    
    with path_to_p_files.joinpath('data_partition.p').open('rb') as fh: partitions = pickle.load(fh)
    new_partition_dict = {}
    
    for partition in partitions:
        new_partition_dict[partition] = []
        IDs = set([x[0] for x in partitions[partition]])
        for ID in IDs:
            observations = [x for x in partitions[partition] if x[0] == ID]
            with path_to_p_files.joinpath(ID+'.p').open('rb') as fp: data = pickle.load(fp)
            for obs in observations:
                if obs[1] in data['stages']: 
                    new_partition_dict[partition].append((ID,obs[1],data['stages'][obs[1]]))
                #assert obs[1] in data['stages'], f"Epoch {obs[1]} not found in {ID}.p!"
                #assert data['stages'][obs[1]] == obs[2], f"For epoch {obs[1]}, stage in .p file is {data['stages'][obs[1]]} but stag
    with path_to_p_files.joinpath('data_partition2.p').open('wb') as fh: pickle.dump(new_partition_dict,fh)


def stage_subsequence_statistics(path_to_xdf_files = '/Users/danielyaeger/Documents/sleep_data/Dennis_Scored_XDF',
                                 path_to_p_files = '/Volumes/Elements/sleep_staging'):
    """
    Returns statistics on the length of the average subsequence of the specified stage
    """
    if not isinstance(path_to_p_files, Path): path_to_p_files = Path(path_to_p_files)
    
    if not isinstance(path_to_xdf_files, Path): path_to_xdf_files = Path(path_to_xdf_files)
    
    # Get IDs of files
    IDs = [f.name.split('.')[0] for f in path_to_p_files.iterdir() if 'X' in f.stem]
    
    seq_dict = {}
    stage_list = [None, 'W', '1', '2', '3', 'R']
    # Get length of consecutive stages from xdf files
    for ID in IDs:
        xdf = load_xdf(str(path_to_xdf_files.joinpath(ID+'.xdf')))
        _, stages = get_stages(xdf)
        stages = np.array(stages)
        for stage in stage_list:
            stage_idx = np.nonzero(stages == stage)[0]
            groups = [len(list(group)) for group in more_itertools.consecutive_groups(stage_idx)]
            if stage not in seq_dict:
                seq_dict[stage] = groups
            else:
                seq_dict[stage].extend(groups)
    
    # Print out length of consecutive stages
    print(seq_dict)
    for stage in seq_dict:
        print(f'Stage: {stage}\tMean Length of Subsequence: {np.mean(np.array(seq_dict[stage]))}\tStd. Dev: {np.std(np.array(seq_dict[stage]))}')
    
    

def plot_check(path_to_pfiles = '/Volumes/Elements/sleep_staging',
               path_to_xdf_files = '/Volumes/TOSHIBA EXT/training_data',
               path_to_channel_list = '/Users/danielyaeger/Documents/Modules/sleep-research-ml/data/supplemental/channel_intersection_list.p'):
    """
    Chooses a random file and a random signal and plots the signal from both the .p and the
    .nkamp files
    """
    if not isinstance(path_to_pfiles, Path): path_to_pfiles = Path(path_to_pfiles)
    
    if not isinstance(path_to_xdf_files, Path): path_to_xdf_files = Path(path_to_xdf_files)
    
    # Get channel intersection list
    with open(str(path_to_channel_list), 'rb') as fh:
        channel_list = pickle.load(fh)
    
    # Pick random channel
    channel = np.random.choice(channel_list)
    
    # Get IDs of files
    IDs = [f.name.split('.')[0] for f in path_to_pfiles.iterdir() if 'X' in f.stem]
    
    # Pick random ID
    ID = np.random.choice(IDs)
    
    # Load data from p file
    with path_to_pfiles.joinpath(ID + '.p').open('rb') as fp:
        pdata = pickle.load(fp)
    
    epochs = list(pdata['stages'].keys())
    
    # Pick random epoch
    epoch = np.random.choice(epochs)
    
    # Print pfile stage information
    print(f"ID: {ID}\nStage from p file: {pdata['stages'][epoch]}")
    
    # Get pfile signal data
    psig_data = pdata['signals'][channel][(epoch-1)*30:epoch*30].flatten()
    psig_fs = len(psig_data)//30
    
    # Load xdf and nkamp files
    path_to_xdf = path_to_xdf_files.joinpath(ID + '.xdf')
    path_to_signal = path_to_xdf_files.joinpath(ID + '.nkamp')
    xdf, signal = load_data(path_to_xdf, path_to_signal)
    
    # Get stage
    _ , stages = get_stages(xdf)
    print(f'Stage from xdf file: {stages[epoch-1]}')
    
    # Get signal
    nsig_data = signal.read_file([channel])
    nsig_data = np.concatenate(nsig_data[channel][(epoch-1)*30:epoch*30])
    nsig_fs = len(nsig_data)//30
    
    #return nsig_data, psig_data

    # Plot data
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(nsig_fs*30)/nsig_fs, nsig_data, label = f'.nkamp: {channel}')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(np.arange(psig_fs*30)/psig_fs, psig_data, label = f'.p: {channel}')
    plt.legend(loc='best')
    plt.show()    

def get_distribution(path_to_pfiles = '/Volumes/Elements/sleep_staging',
                     path_to_xdf_files = '/Users/danielyaeger/Documents/sleep_data/Dennis_Scored_XDF'):
    """
    Returns the distribution of stages in p files and xdf files. Note that the distribution will not be
    exactly the same because the last epoch is truncated in the p file if there is not 30 seconds
    of signal data.
    """
    if not isinstance(path_to_pfiles, Path): path_to_pfiles = Path(path_to_pfiles)
    
    if not isinstance(path_to_xdf_files, Path): path_to_xdf_files = Path(path_to_xdf_files)
    
    p_files = [f for f in path_to_pfiles.iterdir() if 'X' in f.stem]
        
    p_stage_dict = {}
    
    # Get stages from p files
    for p_file in p_files:
        with p_file.open('rb') as fh: data = pickle.load(fh)
        
        stages = data['stages']
        
        for epoch in stages:
            stage = stages[epoch]
            if stage not in p_stage_dict:
                p_stage_dict[stage] = 1
            else:
                p_stage_dict[stage] += 1
    print('Distribution of stages in p files')
    
    for stage in p_stage_dict:
        print(f'\t{stage}: {p_stage_dict[stage]}')
        
    # Collect IDs from p_files
    IDs = [f.name.split('.')[0] for f in p_files]
    
    xdf_stage_dict = {}
    
    # Get stages from xdf files
    for ID in IDs:
        xdf = load_xdf(str(path_to_xdf_files.joinpath(ID+'.xdf')))
        _, stages = get_stages(xdf)
        for stage in stages:
            if stage not in xdf_stage_dict:
                xdf_stage_dict[stage] = 1
            else:
                xdf_stage_dict[stage] += 1
    
    print('Distribution of stages in .xdf files')
    for stage in xdf_stage_dict:
        print(f'\t{stage}: {xdf_stage_dict[stage]}')
    
        
    
def make_stage_dict(path_to_p_files = '/Users/danielyaeger/Documents/sleep_staging'):
    
    if not isinstance(path_to_p_files, Path):
        path_to_p_files = Path(path_to_p_files)
    
    # Build dictionary of stages from .p files
    p_files = [f for f in path_to_p_files.iterdir() if 'X' in f.stem]
    stage_dict = {}
    
    print('Building stage_dict')
    for p_file in p_files:
        ID = p_file.stem.split('_')[0]
        epoch = int(p_file.stem.split('_')[1])
        if ID not in stage_dict:
            stage_dict[ID] = {}
        with p_file.open('rb') as fin: data = pickle.load(fin)
        stage_dict[ID][epoch] = data['stage']
    
    return stage_dict

def check_stage_match(stage_dict, path_to_xdf_files = '/Users/danielyaeger/Documents/sleep_data/Dennis_Scored_XDF'):
    
    if not isinstance(path_to_xdf_files, Path):
        path_to_xdf_files = Path(path_to_xdf_files)
    
    files = [f for f in path_to_xdf_files.iterdir() if f.suffix == '.xdf']
    
    for file in files:
        print('Verifying xdf and p_file have same stage')
        # Build dictionary of stages and IDs from .p files
        ID = file.stem
        if ID in stage_dict.keys():
            xdf = load_xdf(file)
            epochs, stages = get_stages(xdf)
            for i, epoch in enumerate(epochs):
                if stages[i] is not None:
                    try:
                        print(f'xdf stage: {stages[i]}\t stage_dict[ID][epoch]: {stage_dict[ID][epoch]}')
                        assert stage_dict[ID][epoch] == stages[i], f'xdf stage: {stages[i]}\t stage_dict[ID][epoch]: {stage_dict[ID][epoch]}'
                    except KeyError:
                        print(f'only xdf stage found: {stages[i]}')



def plot_random_file(path_to_p_files = '/Users/danielyaeger/Documents/sleep_staging'):
    
    if not isinstance(path_to_p_files, Path):
        path_to_p_files = Path(path_to_p_files)
    
    # Build dictionary of stages from .p files
    p_files = [f for f in path_to_p_files.iterdir() if 'X' in f.stem]
    np.random.shuffle(p_files)
    
    with p_files[0].open('rb') as fh: data = pickle.load(fh)
    
    print(f"The stage is {data['stage']}")
    
    t = np.arange(0,3000)*0.01
    
    plt.plot(t,data['signals']['EOG-L'].flatten())

    
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/danielyaeger/Documents/Modules')
    sys.path.append('/Users/danielyaeger/Documents/Modules/sleep-research-ml/src')
    check_filtering()