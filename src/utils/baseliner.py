#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:16:35 2020

@author: danielyaeger
"""
import numpy as np
from matplotlib import pyplot as plt

def set_baseline(data: np.ndarray, index: int, baseline_length: int = 3, quantile: float = 0.5,
                 sampling_rate: int = 10, step_size: int = 100) -> np.ndarray:
        """ Calculates the baseline of an array given the index and baseline
        length (in seconds). Only considers non-zero vales when calculating the
        baseline.

        Case 1: The first part of the sequence is being analyzed, and the index
        corresponds to a time of less than the value of the baseline_length
        parameter. The first baseline_length seconds of the array are
        used as baseline.

        Case 2: The index corresponds to a time of more than the value of the
        baseline_length parameter. The baseline consists of the last baseline_length
        worth of samples (i.e. baseline_length seconds, or f_s * p_baseline_length
        samples).

        Returns baseline
        """

        if (index - (sampling_rate * baseline_length)) <= 0:
            baseline_data = data[0:int(sampling_rate * baseline_length)]
            baseline = baseline_data[np.nonzero(baseline_data != 0)]
            if len(baseline) == 0:
                baseline = data[data != 0][:int(sampling_rate * baseline_length)]
        else:
            baseline_data = data[index - int(sampling_rate * baseline_length):index]
            baseline = baseline_data[np.nonzero(baseline_data != 0)]
            if len(baseline) == 0:
                baseline = data[data != 0]
        
        assert len(baseline) > 0, f'Baseline has zero length!'
        baseline = np.quantile(a=baseline,q=quantile,axis=0)
        
        # Protect from division by zero
        assert baseline != 0, f'Baseline is equal to zero!'
        return  baseline

def baseline(data: np.ndarray, sampling_rate: int = 10, quantile: float = 0.5,
                  baseline_length: int = 120, step_size: int = 1,
                  replace_zeros: bool = False) -> np.ndarray:
    """Baselines data, by default according to the previous two minutes of data.
    
    INPUTS:
        data: 1 or N-dimensional numpy array
        
    PARAMETERS:
        sampling rate: the sampling rate of the data
        
        quantile: the quantile to use when calculating the baseline (e.g. 0.5 for median)
        
        baseline_length: the length of the baseline in seconds
        
        step_size: the stride length to use when baselining the data.
        
        replace_zeros: if this option is set to True, replaces all zero and negative
        values in the data with the previous value which is greater than zero.
    
    OUTPUTS:
        baselined array
    
    """
    
    if replace_zeros:
        data = fix_zeros(data)
        
    out = np.empty(data.shape)
    for index in range(step_size,data.shape[0]+step_size,step_size):
        baselined = set_baseline(data=data,
                                                   index=index,
                                                   quantile=quantile,
                                                   baseline_length=baseline_length,
                                                   sampling_rate=sampling_rate,
                                                   step_size = step_size)
        out[index-step_size:index] = data[index-step_size:index]/baselined
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
