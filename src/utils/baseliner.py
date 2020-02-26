#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:16:35 2020

@author: danielyaeger
"""
import numpy as np
from matplotlib import pyplot as plt

def set_baseline(data: np.ndarray, labels: np.ndarray, index: int, 
                 baseline_type: str = 'quantile', cut_offs: tuple = (0.01,0.99),
                 baseline_length: int = 3, quantile: float = 0.5,
                 sampling_rate: int = 10) -> tuple:
        """ Calculates the baseline of an array given the index and baseline
        length (in seconds). Only considers non-zero vales when calculating the
        baseline.

        Case 1: The first part of the sequence is being analyzed, and the index
        corresponds to a time of less than the value of the baseline_length
        parameter. The first baseline_length seconds of the array are
        used as baseline.

        Case 2: The index corresponds to a time of more than the value of the
        baseline_length parameter. The baseline consists of the last baseline_length
        worth of samples (i.e. baseline_length seconds, or sampling_rate * 
        baseline_length
        samples).
        
        Inputs:
            data: numpy array of data
            
            labels: numpy array of targets. 0 is assumed to be non-event.
            
            index: integer specifying current index in data
            
            baseline_type: string specifying which type of baseline to calculate
        
        Parameters:
            baseline length: length in seconds over which to calculate baseline.
            
            quantile: which quantile to use if quanile mode is used.
            
            sampling_rate: sampling_rate of data (and labels)
            
            cut_offs: the minimum and maximum to be used when performing min-max
            scaling, in the format (min,max).
        
        Returns:
            baseline, a tuple:
                
                if baseline_type is quantile: tuple contains value of quantile
                
                if baseline_type is min_max: first entry of tuple is 5th quantile
                    of baseline data and second entry is 95th quantile of data.
            

        Returns baseline
        """
        assert baseline_type in ('quantile','min_max'), "Baseline_type must be 'min_max' or 'quantile'!"
        assert cut_offs[1] > cut_offs[0], "Minimum value must be less than maximum value!"
        
        # Handle case where data is not one-dimensional through broadcasting
        if len(data.shape) > 1:
                labels = labels.reshape(-1,1)
        
        if (index - (sampling_rate * baseline_length)) <= 0:
            baseline_data = data[0:int(sampling_rate * baseline_length)]
            baseline_labels = labels[0:int(sampling_rate * baseline_length)]
            baseline = baseline_data[(baseline_data != 0) & (baseline_labels == 0)]

            if len(baseline) == 0:
                baseline = data[(data != 0) & (labels == 0)][:int(sampling_rate * baseline_length)]
        else:
            baseline_data = data[index - int(sampling_rate * baseline_length):index]
            baseline_labels = labels[index - int(sampling_rate * baseline_length):index]
            baseline = baseline_data[(baseline_data != 0) & (baseline_labels == 0)]
            if len(baseline) == 0:
                baseline = data[(data != 0) & (labels == 0)]
        
        assert len(baseline) > 0, f'Baseline has zero length!'
        
        if baseline_type == 'quantile':
            baseline = np.quantile(a=baseline,q=quantile,axis=0)
            assert baseline != 0, f'Baseline is equal to zero!'
            return (baseline,)
        
        elif baseline_type == 'min_max':
            b_min = np.quantile(a=baseline, q=cut_offs[0], axis=0)
            b_max = np.quantile(a=baseline, q=cut_offs[1], axis=0)
            baseline = (b_min,b_max)
        
        return baseline
        

def baseline(data: np.ndarray, labels: np.ndarray, sampling_rate: int = 10, 
             quantile: float = 0.5, baseline_type: str = 'quantile',
             cut_offs: tuple = (0.01,0.99),
             baseline_length: int = 120, step_size: int = 10) -> np.ndarray:
    """Baselines data, by default according to the previous two minutes of data.
    
    INPUTS:
        data: 1 or N-dimensional numpy array
        
        labels: 1-dimensional numpy array
        
        baseline_type: whether to use the quantile of the baseline period or
            the min and max to adjust baseline.
        
    PARAMETERS:
        sampling rate: the sampling rate of the data
        
        quantile: the quantile to use when calculating the baseline (e.g. 0.5 for median)
        
        cut_offs: the minimum and maximum to be used when performing min-max
        scaling, in the format (min,max).
        
        baseline_length: the length of the baseline in seconds
        
        step_size: the stride length to use when baselining the data.
        
        replace_zeros: if this option is set to True, replaces all zero and negative
        values in the data with the previous value which is greater than zero.
    
    OUTPUTS:
        baselined array
    
    """
    assert baseline_type in ('quantile','min_max')
        
    out = np.empty(data.shape)
    for index in range(step_size,data.shape[0]+step_size,step_size):
        baseline = set_baseline(data=data, 
                                 labels = labels,
                                 index=index,
                                 baseline_type = baseline_type,
                                 quantile=quantile,
                                 baseline_length=baseline_length,
                                 sampling_rate=sampling_rate)
        if baseline_type == 'quantile':
            out[index-step_size:index] = data[index-step_size:index]/baseline[0]
        elif baseline_type == 'min_max':
            # b_min is first entry, b_max is second entry
            out[index-step_size:index] = (data[index-step_size:index] - baseline[0])/(baseline[1]-baseline[0])            
    return out

def fix_zeros(data: np.array):
    """
    Changes any zero values to immediately preceding non-zero value.
    Assumes data is organized with channels in columns and rows corresponding
    to different time points.
    """
    
    out = data.copy()
    
    if len(data.shape) == 2:
        # Find indexes of z
        zero_idx = np.nonzero(data <= 0)
        
        for i,row in enumerate(zero_idx[0]):
            column = zero_idx[1][i]
            #Use first nonzero item in column if first item is zero 
            if row == 0:
                out[row,column] = data[:,column][data[:,column] > 0][0]
            
            else:
                out[row,column] = out[row-1,column]
                
    elif len(data.shape) == 1:
        zero_idx = np.nonzero(data <= 0)[0]
        
        for i in zero_idx:
            #Use first nonzero item in array if first entry is zero 
            if i == 0:
                out[i] = data[data > 0][0]
            else:
                out[i] = out[i-1]
    
    assert np.all(out > 0), f"Output array contains values less than or equal to zero: \n{np.nonzero(out <= 0)}!"
    return out
        
def baseline_sine_wave():
    "Test function to look at what min-max scaling a sine wave looks like"
    # Create time from sine wave
    fs = 10
    time = np.arange(10000)*(1/fs)
    y = np.sin(time)
    
    baselined = baseline(data = y, labels = np.zeros(len(y)), baseline_type = 'min_max')
    
    plt.hist(baselined)
    plt.title('Baselined Sine wave')

def test_baseliner():
    "Plots data with and without baselining"
    # Create random array
    x = np.random.random((30000,2))

    # Add 10 to the second column
    x[:,1] += 100

    # Create changing baseline_length
    for i in range(0,100):
        x[i*300:(i+1)*300,:] += i
    
    # Create random zeros
    zeros_one = np.random.randint(0,30000,2000)
    zeros_two = np.random.randint(0,30000,2000)
    x[zeros_one,0] = 0
    x[zeros_two,1] = 0
    
    # Baseline
    base = baseline(data=x,sampling_rate=10,quantile=0.5,baseline_length=3,step_size=300)

    # Plot
    plt.subplot(411)
    plt.plot(x[:,0],label='First column')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(base[:,0],label='First column baselined')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(x[:,1],label='Second column')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(base[:,1],label='First column baselined')
    plt.legend(loc='best')

if __name__ == "__main__":
    baseline_sine_wave()
