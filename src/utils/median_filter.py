#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:57:17 2020

@author: danielyaeger
"""
import numpy as np
from matplotlib import pyplot as plt
from biosppy.utils import ReturnTuple

def load_ecg(ID = 'XAXVDJYND80Q4Q0', input_path = '/Volumes/TOSHIBA EXT/training_data'): 
   xdf, signal = load_data(input_path + '/' + ID + '.xdf', input_path + '/' + ID + '.nkamp') 
   ecg = signal.read_file('ECG') 
   return ecg['ECG'].ravel()            

class MedianFilter():
    """
     PARAMETERS:
        lower_bound: shortest physiological RR interval in seconds, typically
        0.4 seconds, or 150 BPM
        
        upper_bound: longest physiological RR interval in seconds, typically
        2 seconds, or 30 BPM
        
        sampling_rate: sampling rate in samples per second, typically 200
        
        half_window_size: samples to consider when adjusting minimum and maximum
        intervals
    """
    
    def  __init__(self, filter_lower: bool = True, filter_upper: bool = False,
                  upper_bound: float = 2, lower_bound: float = 0.3, 
                  input_sampling_rate: int = 200, output_sampling_rate: int = 10, 
                  half_window_size = 5):
        
        self.filter_lower = filter_lower
        self.filter_upper = filter_upper
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.output_sampling_rate = output_sampling_rate
        self.input_sampling_rate = input_sampling_rate
        self.half_window_size = half_window_size

    def generate_hr_time_series(self, ecg_data: ReturnTuple, ecg_sig: np.array, end_time: int, show = False):
        
        """Takes in ecg data (a ReturnTuple object) and returns a numpy array with
        ones at R wave peaks and zero elsewhere, at the output_sampling_rate and
        of length end_time X output_sampling_rate
        
        INPUT:
            ecg_data: a biosippy.utils ReturnTuple with 'rpeaks' field
            
            end_time: end time of data in ecg_data
            
            show: whether to plot sampled time series and original time series
        
        RETURNS:
            
            tuple (rr_peaks, ts, out)
            
            rr_peaks: numpy array with the timing of heart beats
            
            ts: time series at output_sampling_rate from 0 to end_time - output_sampling_rate
            
            out: numpy array at output_sampling_rate, with ones in bins corresponding
            to heart beat and zeros elsewhere
            
        """
        out = np.zeros(int(end_time*self.output_sampling_rate))
        ts = np.arange(int(end_time*self.output_sampling_rate))*(1/self.output_sampling_rate)
        
        rpeaks = ecg_data['rpeaks'].copy()
        
        # Get intervals
        intervals = np.diff(rpeaks)
        
        # Median filter RR intervals
        if self.filter_lower or self.filter_upper:
            intervals, rpeaks = self.run_median_filter(intervals, rpeaks)
        
        # Make sure intervals is of sufficient length
        assert len(intervals) >= 1, f'Fewer than 1 RR intervals! Only {len(intervals)} RR intervals found!'
        
        rr_peaks = np.zeros(len(intervals) + 1)
        
        # Convert corrected intervals to heart beat time series
        for i in range(len(rr_peaks)):
            if i == 0:
                rr_peaks[0] = rpeaks[0]
            else:
                rr_peaks[i] = rr_peaks[i-1] + intervals[i-1]
        
        # If no filtering, signals should be identical
        if not (self.filter_lower or self.filter_upper):
            assert (rr_peaks == rpeaks).all(), "rr_peaks and ecg_data['rpeaks'] are not equal!"
        
        # Convert time of peaks to time series with R peaks
        for i,time in enumerate(rr_peaks):
            if time <= end_time:
                # index is greatest index less than time
                idx = np.nonzero(ts <= time)[0][-1]
                if not (self.filter_lower or self.filter_upper):
                    out[idx] = np.abs(rpeaks[i])
                else:
                    out[idx] = 1
        
        if show:
            plt.subplot(2,1,1)
            plt.plot(ecg_sig, label = 'ECG')
            plt.legend(loc='best')
            plt.subplot(2,1,2)
            plt.plot(ts,out, label = f'R peaks @ {self.output_sampling_rate} Hz')
            plt.legend(loc='best')
            plt.show()
            
        return out
    
    
    def run_median_filter(self,intervals: np.array, rpeaks: np.array):
        """Calls the median filter function to filter input array until arrays
        stop changing.
        
        INPUTS:
            intervals: array of RR intervals (in seconds)
        
        RETURNS:
            corrected intervals as numpy array
        """
        assert len(intervals) > 0, 'Intervals has length zero!'
        
        if self.filter_lower:
            corrected, rpeaks = self.lower_median_filter(intervals=intervals, rpeaks=rpeaks)
        if self.filter_upper:
            if self.filter_lower:
                corrected = self.upper_median_filter(intervals=corrected)
            else:
                corrected = self.upper_median_filter(intervals=intervals)
        
        # If neither filter upper or filter lower, just return intervals
        if (not self.filter_lower) and (not self.filter_upper):
            return intervals
        
        while not self.test_for_equality(corrected,intervals):
            intervals = corrected.copy()
            del corrected
            if self.filter_lower:
                corrected, rpeaks = self.lower_median_filter(intervals=intervals, rpeaks=rpeaks)
            if self.filter_upper:
                if self.filter_lower:
                    corrected = self.upper_median_filter(intervals=corrected)
                else:
                    corrected = self.upper_median_filter(intervals=intervals)
        
        if  self.filter_lower:
            assert np.all(corrected > self.lower_bound), f'Values in filtered array less than lower bound of {self.lower_bound}: {corrected[corrected <= self.lower_bound]} at indices {np.nonzero(corrected <= self.lower_bound)[0]}!'
        
        if self.filter_upper:
            assert np.all(corrected < self.upper_bound), f'Values in filtered array greater than upper bound of {self.upper_bound}: {corrected[corrected >= self.upper_bound]} at indices {np.nonzero(corrected >= self.upper_bound)[0]}!'
        
        assert len(rpeaks) == (len(corrected) + 1), f"length of corrected ({len(corrected)}) not equal to length of corrected ({len(corrected)})!"
        return corrected, rpeaks
    
    def test_for_equality(self,a,b):
        """Returns True if two arrays are equal. Otherwise returns False.
        """
        if len(a) != len(b):
            return False
        else:
            return np.all(a==b)
    
    def find_median_for_window(self, intervals: np.array, center_idx: int, half_window_size: int = 5):
        """Finds the median for the window with center index of center_idx and size of
        2*half_window_size + 1
        
        INPUTS:
            intervals: numpy array
            center_idx: index of center interval
        
        PARAMETERS:
            half_window_size: window size integer division by 2 and plus 1
        """ 
        median = 0
        
        half_window_size = half_window_size
        
        while not (self.lower_bound < median < self.upper_bound):
            
            if (half_window_size + 1) <= center_idx <= (len(intervals) - 1 - half_window_size):
                window = intervals[center_idx - half_window_size - 1:center_idx + half_window_size]
                assert len(window) == (2*half_window_size + 1), f'Window size is {len(window)}, center index is {center_idx}, but half_window_size is {half_window_size}!'
            
            elif center_idx >= (len(intervals) - 1 - half_window_size):
                window = intervals[center_idx - half_window_size - 1:]
            elif center_idx <= (half_window_size + 1):
                window = intervals[:center_idx + half_window_size]
            
            assert len(window) > 0, f'Window size is zero with {center_idx} and {half_window_size}!'
            
            median =  np.median(window)
            
            half_window_size += 1
        
        return median
    
    def lower_median_filter(self,intervals: np.array, rpeaks: np.array):
        """ Implements the lower median filter as described in Chen, Song, & Zhang, 2015
        
        INPUTS: 
            intervals: array of RR intervals
        
        RETURNS:
            corrected intervals"""
        out = list(intervals.copy())
        rpeaks = list(rpeaks.copy())
        
        # Look at lowest bound first
        # Find minimum position
        min_position = np.argmin(intervals)
        
        if intervals[min_position] <= self.lower_bound:
            if 0 < min_position <= len(intervals) - 2:
                median = self.find_median_for_window(intervals=intervals, 
                                                center_idx=min_position, 
                                                half_window_size=self.half_window_size)
                
                d_1 = np.abs(intervals[min_position] + min(intervals[min_position-1], 
                                   intervals[min_position+1]) - median)
                
                d_2 = np.abs(2*(0.5*(intervals[min_position] + max(intervals[min_position-1], 
                                   intervals[min_position+1])) - median))
                
                # if d_min_2 < d_min_1, average value in min_position with greatest neighbor
                if d_2 < d_1:
                    if intervals[min_position+1] >= intervals[min_position-1]:
                        out[min_position+1] = np.mean([intervals[min_position+1], intervals[min_position]])
                        out[min_position] = out[min_position+1]
                    else:
                        out[min_position-1] = np.mean([intervals[min_position-1], intervals[min_position]])
                        out[min_position] = out[min_position-1]
                
                # Remove lowest interval
                else:
                    if intervals[min_position+1] <= intervals[min_position-1]:
                        out[min_position+1] = intervals[min_position+1] + intervals[min_position]
                    else:
                        out[min_position-1] = intervals[min_position-1] + intervals[min_position]
                    out.pop(min_position)
                    rpeaks.pop(min_position+1)
                
            elif min_position == 0:
                out[min_position+1] = np.mean([intervals[min_position+1], intervals[min_position]])
                out[min_position] = out[min_position+1]
            
            elif min_position == len(intervals) - 1:
                out[min_position-1] = np.mean([intervals[min_position-1], intervals[min_position]])
                out[min_position] = out[min_position-1]
                
        # Update intervals to account for removing interval
        return  np.array(out), np.array(rpeaks)
    
    def upper_median_filter(self,intervals: np.array):
        """ Implements the upper median filter as described in Chen, Song, & Zhang, 2015
        
        INPUTS: 
            intervals: array of RR intervals
        
        RETURNS:
            corrected intervals"""
        
        out = list(intervals.copy())
        max_position = np.argmax(intervals)
        
        if intervals[max_position] >= self.upper_bound:
            # Variable to check whether max value was changed
            changed_max = False
            if 0 < max_position <= len(intervals) - 2:
                median = self.find_median_for_window(intervals=intervals, 
                                                center_idx=max_position, 
                                                half_window_size=self.half_window_size)
                
                k = max(intervals[max_position]/median, 2)
                d_1 = np.abs(k*((intervals[max_position]/k) - median))
                
                d_2 = np.abs(2*(0.5*(intervals[max_position] + min(intervals[max_position-1], 
                                   intervals[max_position+1])) - median))
                if d_1 < d_2:
                    #print(f'\t\tintervals[max_position] = {intervals[max_position]}')
                    #print(f'Value of median: {median}')
                    interp_rr = intervals[max_position]/int(k)
                    #print(f'Value of k: {k}')
                    for i in range(int(k)):
                        #print(f'\tAdding {i}th interval of size {interp_rr}')
                        #print(out)
                        if i == 0:
                            out[max_position] = interp_rr
                        else:
                            out.insert(max_position, interp_rr)
                    changed_max = True
                else:
                    if intervals[max_position+1] <= intervals[max_position-1]:
                        out[max_position+1] = np.mean([intervals[max_position+1], intervals[max_position]])
                        out[max_position] = out[max_position+1]
                    else:
                        out[max_position-1] = np.mean([intervals[max_position-1], intervals[max_position]])
                        out[max_position] = out[max_position-1]
                    changed_max = True
            elif max_position == 0:
                median = self.find_median_for_window(intervals=intervals, 
                                                center_idx=max_position, 
                                                half_window_size=self.half_window_size)
                k = max(intervals[max_position]/median, 2)
                d_1 = np.abs(k*((intervals[max_position]/k) - median))
                d_2 = np.abs(2*(0.5*(intervals[max_position] + intervals[max_position+1]) - median))
                if d_1 < d_2:
                    #print(f'\t\tintervals[max_position] = {intervals[max_position]}')
                    #print(f'Value of median: {median}')
                    interp_rr = intervals[max_position]/int(k)
                    #print(f'Value of k: {k}')
                    for i in range(int(k)):
                        #print(f'\tAdding {i}th interval of size {interp_rr}')
                        #print(out)
                        if i == 0:
                            out[max_position] = interp_rr
                        else:
                            out.insert(max_position, interp_rr)
                    changed_max = True
                else:
                    out[max_position+1] = np.mean([intervals[max_position+1], intervals[max_position]])
                    out[max_position] = out[max_position+1]
            
            elif max_position == (len(intervals) - 1):
                median = self.find_median_for_window(intervals=intervals, 
                                                center_idx=max_position, 
                                                half_window_size=self.half_window_size)
                k = max(intervals[max_position]/median, 2)
                d_1 = np.abs(k*((intervals[max_position]/k) - median))
                d_2 = np.abs(2*(0.5*(intervals[max_position] + intervals[max_position-1]) - median))
                if d_1 < d_2:
                    #print(f'\t\tintervals[max_position] = {intervals[max_position]}')
                    #print(f'Value of median: {median}')
                    interp_rr = intervals[max_position]/int(k)
                    #print(f'Value of k: {k}')
                    for i in range(int(k)):
                        #print(f'\tAdding {i}th interval of size {interp_rr}')
                        #print(out)
                        if i == 0:
                            out[max_position] = interp_rr
                        else:
                            out.insert(max_position, interp_rr)
                    changed_max = True
                else:
                    out[max_position-1] = np.mean([intervals[max_position-1], intervals[max_position]])
                    out[max_position] = out[max_position-1]
                
                
            assert changed_max, print(f"{intervals[max_position]} still {out[max_position]}")
   
        return np.array(out)
        

    
    
#if __name__ == "__main__":
#    test_median_filter_real_data(end_time=10000)

    
            
            
            
            
        
        