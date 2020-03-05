#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:33:09 2020

@author: danielyaeger
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
from mpl_toolkits.mplot3d import Axes3D

def plot_signal_over_different_sleepers(channels = ['SpO2', 'Airflow', 'P-Flo'],
                     data_path = '/Users/danielyaeger/Documents/raw_baselined_v2',
                     label = 0, windowsize = 600) -> (list, dict):
    
    if not isinstance(data_path,Path): data_path = Path(data_path)
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as fin: targets = pickle.load(fin)
    
    ID_list = [f.name.split('.')[0] for f in data_path.iterdir() if f.name.startswith('X')]
    
    data = {}
    for ID in ID_list:
        data[ID] = np.load(str(data_path.joinpath(ID + '.npy')))
    
    with data_path.joinpath('channel_list.p').open('rb') as fh: channel_list = pickle.load(fh)
    
    samples = {channel: [] for channel in channels}
    counter = 0
    for ID in ID_list:
        added_signal = False
        while not added_signal:
            time = np.random.choice(np.arange(len(targets[ID])-windowsize))
            if (targets[ID][time:time + windowsize]  == label).all():
                added_signal = True
                for channel in channels:
                    samples[channel].append(data[ID][time:time + windowsize, channel_list.index(channel)])
    
    #return (ID_list, samples)

    for i, channel in enumerate(channels):
        plt.subplot(len(channels),1,i+1)
        for j,ID in enumerate(ID_list):
            plt.plot(samples[channel][j],label=f'{channel}_{ID}')
            plt.legend()  
    
    
    
    

def get_signal_over_time(channels = ['SpO2', 'Airflow', 'P-Flo'],
                     ID = 'XAXVDJYND7Q6JTK',
                     data_path = '/Users/danielyaeger/Documents/raw_baselined_v2',
                     label = 0, number_required = 10,
                     windowsize = 600, counter_max = 10e6) -> (list, dict):
    "Randomly samples data to show window of data over time"
    
    if not isinstance(data_path,Path): data_path = Path(data_path)
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as fin: targets = pickle.load(fin)[ID]
    
    data = np.load(str(data_path.joinpath(ID + '.npy')))
    
    # Lump apnea/hypopneas together
    targets[targets > 0] = 1
    
    if label == 1:
        assert (targets == 1).any(), "No apnea/hypopnea for ID!"
    
    with data_path.joinpath('channel_list.p').open('rb') as fh: channel_list = pickle.load(fh)
    
    samples = {channel: [] for channel in channels}
    start_times = []
    counter = 0
    while len(start_times) < number_required:
        time = np.random.choice(np.arange(len(targets)-windowsize))
        if (targets[time:time + windowsize]  == label).all():
            start_times.append(time)
            for channel in channels:
                samples[channel].append(data[time:time + windowsize, channel_list.index(channel)])
        counter += 1
        if counter > counter_max: break
    
    return (start_times, samples)

def plot_random_times(start_times: list, samples: dict):
    "Randomly plots number_to_plot windows from start_times"

    
    # Divide by sampling rate to get time
    times = np.array(start_times)/(10)
    print(times)
    
    selected_ch_dict = {channel: [] for channel in list(samples.keys())}
    
    for j,_ in enumerate(start_times):
        for channel in list(samples.keys()):
            selected_ch_dict[channel].append(samples[channel][j])
    
    for i, channel in enumerate(list(samples.keys())):
        plt.subplot(len(list(samples.keys()))+1,1,i+1)
        for j in np.argsort(np.array(start_times)):
            plt.plot(selected_ch_dict[channel][j],label=f'{channel}_{times[j]}')
            plt.legend()    

def get_class_distribution(channel = 'SpO2',
                            file_list = None,
                            data_path = '/Users/danielyaeger/Documents/raw_baselined_v2'):
    """ Plots class-specific distributions
    """
    if not isinstance(data_path,Path): data_path = Path(data_path)
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as fin: targets = pickle.load(fin)
    
    with data_path.joinpath('channel_list.p').open('rb') as fh: channel_list = pickle.load(fh)
    index = channel_list.index(channel)
    
    feature_dict = {0: [], 1: []}
    
    if file_list is None:
        ID_list = [f.name.split('.')[0] for f in data_path.iterdir() if f.name.startswith('X')]
    else:
        ID_list = sorted(file_list)
    
    
    for ID in ID_list:
        print(f'Fetching data for {ID}')
        data = np.load(str(data_path.joinpath(ID + '.npy')))
        if data.shape[-1] != len(channel_list):
            continue
        data = data[:,index]
        targets[ID][targets[ID] > 0] = 1
        for label in feature_dict:
            idx = np.nonzero(targets[ID]==label)[0]
            feature_dict[label].extend(list(data[idx]))
    return feature_dict

def get_signal_window(channels = ['SpO2', 'Airflow', 'P-Flo', 'Chest', 'Abdomen'],
                     data_path = '/Users/danielyaeger/Documents/new_raw_baselined'):
    
    if not isinstance(data_path,Path): data_path = Path(data_path)
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as fin: targets = pickle.load(fin)
    
    with data_path.joinpath('channel_list.p').open('rb') as fh: channel_list = pickle.load(fh)
    
    template = np.concatenate((np.zeros(50),np.ones(100)))
    
    samples = {channel: [] for channel in channels}
    
    ID_list = [f.name.split('.')[0] for f in data_path.iterdir() if f.name.startswith('X')]
     
    for ID in ID_list:
        data = np.load(str(data_path.joinpath(ID + '.npy')))
        for i in range(len(targets[ID]) - len(template)):
            if (targets[ID][i:i+ len(template)] == template).all():
                for channel in channels:
                    index = channel_list.index(channel)
                    samples[channel].append(data[i:i+ len(template),index])
    
    for channel in samples:
        samples[channel] = np.array(samples[channel])
    
    summary = {}
    for channel in samples:
        summary[channel] = {}
        summary[channel]['mean'] = np.mean(samples[channel], axis=0)
        summary[channel]['std'] = (np.std(samples[channel], axis=0))**2
    
    for i, channel in enumerate(channels):
        plt.subplot(len(channels)+1,1,i+1)
        y = summary[channel]['mean']
        x = np.arange(len(template))
        error = summary[channel]['std']
        plt.plot(x,y,color='blue')
        plt.fill_between(x, y-error, y+error,color = 'blue')
        plt.plot(x,y,label=channel,color='orange')
        plt.legend()
            
    
    plt.subplot(len(channels)+1,1,i+2)
    plt.plot(np.arange(len(template)), template, color = 'red', label = 'Apnea/Hypopnea')
    plt.legend()
    plt.xlabel('Time')
        
    
    
    
    
    
    
    
    
    
def plot_class_distribution(feature_dict,channel_name, filter_by_quantile = True): #,filter_high,filter_low):
    # Normalize
    kwargs = dict(alpha=0.5, bins=1000, density=True, stacked=True)
    
    # Filter out extreme values
    for label in feature_dict:
        if filter_by_quantile:
            high = np.quantile(feature_dict[label], 0.99)
            low = np.quantile(feature_dict[label], 0.01)
            x = feature_dict[label]
            x = np.array(x)
            x = x[(x > low) & (x < high)]
            feature_dict[label] = x
    
    f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    ax1.hist(feature_dict[0],**kwargs,label='None')
    ax1.legend()
    ax2.hist(feature_dict[1],**kwargs,label='Apnea/Hypopnea')
    ax2.legend()

    
    
def plot_signals_and_apneas(channel = 'Chest_energy',
                            file = 'XAXVDJYND6ZBY93.npy',
                            data_path = '/Users/danielyaeger/Documents/apnea_data_v2'):
    
    if not isinstance(data_path,Path): data_path = Path(data_path)
    
    with data_path.joinpath('apnea_hypopnea_targets.p').open('rb') as fin: targets = pickle.load(fin)
    
    with data_path.joinpath('channel_list.p').open('rb') as fh: channel_list = pickle.load(fh)
    index = channel_list.index(channel)
    data = np.load(str(data_path.joinpath(file)))[:,index]
    labels = targets[file.split('.')[0]]
    labels[np.nonzero(labels>0)[0]] = 1.5
    labels[np.nonzero(labels==0)[0]] = 1.0
    plt.plot(data,label = f'{channel}')
    plt.plot(labels, label = f'label')
    plt.legend(loc='best')

def plot_3_d_signals(channels = ['SpO2', 'Airflow', 'P-Flo'],
                     data_path = '/Users/danielyaeger/Documents/new_raw_baselined'):
    none_dict = {}
    ah_dict = {}
    
    for channel in channels:
        feature_dict = get_class_distribution(channel)
        for label in feature_dict:
            if label == 0:
                none_dict[channel] = feature_dict[label]
            elif label == 1:
                ah_dict[channel] = feature_dict[label]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(none_dict['SpO2'], none_dict['Airflow'], none_dict['P-Flo'], 'o')
    ax.plot(ah_dict['SpO2'], ah_dict['Airflow'], ah_dict['P-Flo'], 'o')
    ax.set_xlabel('SpO2')
    ax.set_ylabel('Airflow')
    ax.set_zlabel('P-Flo')
    ax.legend(['None','Apnea/Hypopnea'])


if __name__ == '__main__':
    #plot_signals_and_apneas()
    #channel_name = 'P-Flo'
    #feature_dict = get_class_distribution(channel_name)
    #plot_class_distribution(feature_dict, channel_name)
    #get_signal_window()
    #tup = get_signal_over_time()
    #plot_random_times(start_times=tup[0], samples=tup[1])
    plot_signal_over_different_sleepers()

    
    
    
    
    