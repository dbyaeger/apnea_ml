#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:13:41 2020

@author: danielyaeger
"""
import numpy as np
import collections

def collapse_events(a_events: np.ndarray, h_events: np.ndarray) -> np.ndarray:
    """ Collapses apnea and hypoapnea events into a single track. When events overlap,
    apnea events are given precedence over hypoapnea events.
    
    INPUT:  a_events: larray of apnea events, 
            h_events: array of hypoapnea events,
           
    OUTPUT: list of numpy array with apnea and hypoapnea events collapsed into a
            single track. Apnea events are given priority over hypoapnea events.
            Also returns the number of events where there was overlap of apnea
            and hypoapnea events.
    
    """
    assert type(a_events) == type(h_events) == np.ndarray, f"Both a_events and h_events must be numpy arrays, not {type(a_events)} and {type(h_events)}!"
    
    a_events[np.nonzero(a_events)] = 2
    single_track = np.maximum(a_events, h_events)
    overlap_counter = sum(h_events == 1) - sum(single_track == 1)
    return single_track, overlap_counter

def adjust_event_times(time_list: list, end_time: int,
                       start_time: int = 0) -> list:
    """ Adjusts the start time of events to be no greater than the
    start time and the end time of events to be no greater than the
    end time. This is required because some annotated events end after
    the end time.
        
        new start time = max(event start time, start time)
        new end time = min(event end time, end time)
    
    INPUT: list of tuples in the format 
        (event start time, event end time, event type)
    
    OUTPUT: list of tuples
    """
    return [(max(t[0], start_time), min(t[1], end_time), t[2]) for t in time_list]


def convert_to_idx(time: float, end_time: int, start_time: int=0, 
                       f_s: int=10) -> int:
    """Converts a time to an index. Returns the index relative
    to the start of the signal.
    """

    time -= start_time

    seconds_idx = (time // 1)*f_s

    frac_idx = np.floor((time % 1)*f_s)

    return min(int(seconds_idx +  frac_idx), (end_time - start_time)*f_s - 1)

def sequence_builder(groups: list, length: int) -> np.ndarray:
    """Builds sequences of 0 and 1's given a list of indices in consecutive
    order. Returns a numpy array.

    INPUT: groups, a list of arrays of indices in consecutive order,
    the length of the array to be returned.

    OUTPUT: 1-D numpy array where index entries are ones and all other values
    are zero.
    """
    if len(groups) > 0:
        assert (length - 1) >= np.max([np.max(l) for l in groups]), f"Length must be at least as big as maximum index! Length: {length} and max idx: {np.max([np.max(l) for l in groups])}"
    
    out = np.zeros(length)
    if len(groups) > 0:
        idx = np.concatenate(groups)
        out[idx] = 1
    return out

def convert_to_sequence(time_list: list, end_time: float, 
                        start_time: float = 0, sampling_rate: int = 10) -> np.array:
    """Converts list of events in the format:
            [(start_time, end_time, event_type), ...].
            
            Assumes there are only two types of events: 'A' and 'H'
            
            Returns a numpy array in which each type of event is indicated
            by a unique number for each sample. Apneas are indicated by a 2
            and hypoapneas by a 1. 0 indicates no event.
  
    """
    #sort list
    time_list.sort()
    
    # filter out events that occurred before end_time
    time_list = [event for event in time_list if event[0] <= end_time]
    
    event_times = adjust_event_times(time_list = time_list,
                                                  start_time = start_time,
                                                  end_time = end_time)
    a_idx, h_idx = [], []
    for event in event_times:
        if event[-1] == 'A':
            # Convert event times into indexes relative to start of REM
            a_idx.append(list(np.arange(start = convert_to_idx(time = event[0], 
                                                               start_time = start_time, 
                                                               end_time = end_time,
                                                               f_s = sampling_rate),
                                        stop = convert_to_idx(time = event[1], 
                                                              start_time = start_time, 
                                                              end_time = end_time,
                                                              f_s = sampling_rate) + 1)))
        elif event[-1] == 'H':
            h_idx.append(list(np.arange(start = convert_to_idx(time = event[0], 
                                                               start_time = start_time, 
                                                               end_time = end_time,
                                                               f_s = sampling_rate),
                                        stop = convert_to_idx(time = event[1], 
                                                              start_time = start_time, 
                                                              end_time = end_time,
                                                              f_s = sampling_rate) + 1)))
    length_in_idx = int((end_time - start_time) * sampling_rate)
    a_seq = sequence_builder(groups = a_idx, length = length_in_idx)
    h_seq = sequence_builder(groups = h_idx, length = length_in_idx)
    combined_array, _ = collapse_events(a_events = a_seq, h_events = h_seq)
    
    return combined_array



