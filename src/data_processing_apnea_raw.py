import os
import sys
import time
import pickle
from biosppy.signals.ecg import ecg
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from collections import defaultdict, Counter
from src.utils.dsp import design_filter, filter_and_resample, lowpass
from resampy import resample
import openxdf
from analysis import Analyzer2
from src.utils.median_filter import MedianFilter
from src.utils.open_xdf_helpers import (load_data, 
                                    select_rem_epochs, 
                                    select_rswa_events, 
                                    get_cpap_start_epoch,
                                    get_notes,
                                    parse_notes)
from src.utils.list_to_seq import convert_to_sequence
from src.utils.baseliner import baseline
from sklearn.preprocessing import Normalizer

def run():
    import sys
    sys.path.append('/Users/danielyaeger/Documents/Modules')
    sys.path.append('/Users/danielyaeger/Documents/Modules/sleep-research-ml/src')


class DataProcessorApnea:

    def __init__(self, input_path = '/Volumes/TOSHIBA EXT/training_data',
                 output_path = '/Users/danielyaeger/Documents/raw_baselined_v2',
                 channel_list = ['Abdomen','Chest','Airflow','SpO2','P-Flo','Snore', 'ECG'],
                 path_to_polysmith_db = '/Users/danielyaeger/Documents/Modules/sleep-research-ml/data/supplemental/Polysmith_DataBase_ML_forPhil.csv',
                 min_time_hours=4, min_rem_epochs=10, sampling_rate = 10, 
                 ecg_filter_upper = False,
                 ecg_filter_lower = False, ecg_lower_bound = 0.3, 
                 ecg_upper_bound = 2, baseline = True, normalize = False,
                 ID_list = None, hr = True):
        """
        Takes in xdf and nkamp files corresponding to one complete study and writes
        to disk individual signal files as .npy files with the channels
        in the order of channel_list, which is also saved in the output directory
        as a .p file.
        
        INPUTS:
            input_path: path to directory where .nkamp/.xdf files live
            
            output_path: path to directory where output whould be saved
            
            channel_list: seleted channels to be saved. If set to None,
                all of the channels in the channel_intersection list will be used.
            
            path_to_channel_list: path to channel_intersection_list.p
            
            path_to_polysmith_db: path to /Polysmith_DataBase_ML_forPhil.csv
            
            min_time_hours: minimum total time for a study to be included
            
            rem_epochs: minimum number of REM epochs for a study to be included
                       
            ID_list: provide a list of IDs to limit data processing to the select
                IDs found in the input_path.
            
            ecg_filter_upper: whether to apply median filter to rr peaks in ecg
                signal exceeding ecg_upper_bound value.
            
            ecg_filter_lower: whether to apply median filter to rr peaks in ecg
                signal exceeding ecg_lower_bound value.
            
            ecg_upper_bound: maximum physiological interval of rr peaks
            
            ecg_lower_bound: minimum physiological interval of rr peaks
                
        
        OUTPUTS:
            
            channel_list.p: A pickled list with the channels from the .nkamp 
            file in the order corresponding to the rows of the numpy arrays.
            
            stage_dict.p: A pickled dictionary keyed by ID. For each ID, there
            is a dictionary keyed by epoch for which the value is the human-scored
            (by default according to Dennis) sleep stage.
            
            numpy arrays named as <ID>.npy. The rows contain the channels in the
            order of channel_list
            
            apnea_hypoapnea_targets: A pickled dictionary keyed by ID with the
            value corresponding to the array of output targets (array of 0's, 
            1's (Hypoapnea), and 2's (Apnea)').

        """
        print(f"processing data from {input_path}")
        self.path = Path(input_path)
        assert self.path.is_dir(), f"{input_path} is not a valid path"
            
        
        self.channel_list = sorted(channel_list)
        self.n_channels = len(self.channel_list)
        
        self.apnea_hypoapnea_targets = {}
            
        self.min_time_hours = min_time_hours
        self.min_rem_epochs = min_rem_epochs
        self.fs = sampling_rate
        self.ecg_filter_upper = ecg_filter_upper
        self.ecg_filter_lower = ecg_filter_lower
        self.ecg_lower_bound = ecg_lower_bound
        self.ecg_upper_bound = ecg_upper_bound
        self.baseline = baseline
        self.normalize = normalize
        self.hr = hr
        
        assert not (self.baseline and self.normalize), "Both normalize and baseline options should not be set to True!"
        
        
        # Collect study metadata
        if not isinstance(path_to_polysmith_db,Path): path_to_polysmith_db = Path(path_to_polysmith_db)
        self.study_meta_data = pd.read_csv(path_to_polysmith_db, usecols=['Sex','Age (yrs)','RecType','RecordNo'])
        self.psg_studies = list(self.study_meta_data[self.study_meta_data['RecType'] == 'PSG']['RecordNo'].values)
        self.split_studies = list(self.study_meta_data[self.study_meta_data['RecType'].isin(['SPLIT'])]['RecordNo'].values)
            
        # define path to write out processed data and create dir if it does not exist
        if not isinstance(output_path,Path): output_path = Path(output_path)
        self.output_data_path = output_path
        if not self.output_data_path.is_dir():
            self.output_data_path.mkdir()
        
        # write 
        with self.output_data_path.joinpath("channel_list.p").open("wb") as fl:
            pickle.dump(self.channel_list, fl)

        self.files = [f.parent.joinpath(f.stem) for f in self.path.iterdir() 
                      if f.suffix == '.xdf']
        
        self.ID_list = ID_list
        
        self.stage_dict = {}
    

    #@timeout_decorator.timeout(300)
    def _process_helper(self, f):
        EPOCH_LEN = 30
        ID = f.stem
       
        # Check if study is split or psg type
        assert np.any([ID in self.psg_studies, ID in self.split_studies]), f"{ID} is of {self.study_meta_data[self.study_meta_data['RecordNo'] == ID]['RecType'].values[0]} type"
        
        # load xdf and signal data and pull dataframe
        xdf, signal = load_data(str(f) + ".xdf", str(f) + ".nkamp")
        df = xdf.dataframe()
        df = df[df["Scorer"] == "Dennis"]
        df_rem = df[df["Stage"]=='R']
        epochs = list(df["EpochNumber"])
        stages = list(df["Stage"])
        rem_epochs = np.array(df_rem["EpochNumber"].unique(), dtype = np.int32)
        final_epoch = np.array(epochs).max()
        print('\tRetrieving signals')
        signal_dict = signal.read_file(self.channel_list)
        
        # Ensure min rem epochs are present
        if len(rem_epochs) < self.min_rem_epochs:
            e = "EXIT: min number of REM epochs not present in study"
            print(e)
            self.history[ID] = e
            return
        
        # Ensure study is at least 4 hours
        if len(epochs)*30 < self.min_time_hours*60*60:
            e = "EXIT: study less than 4 hours"
            print(e)
            self.history[ID] = e
            return
        
        # Check if last epoch complete
        end_time = signal_dict[self.channel_list[0]].shape[0]
        print(f"\tSpO2 sampling rate = {signal_dict['SpO2'].shape[-1]}")
        
        # split study, restrict to 30 minutes before cpap machine turned on
        if ID in self.split_studies:
            field_cpap_start = get_cpap_start_epoch(xdf)
            notes = get_notes(str(self.path.joinpath(ID + '.xdf')),xdf)
            notes_cpap_start = parse_notes(notes)
            if field_cpap_start is None and notes_cpap_start is not None:
                final_epoch = notes_cpap_start - 60
            elif field_cpap_start is not None and notes_cpap_start is None:
                final_epoch = field_cpap_start - 60
            elif field_cpap_start is not None and notes_cpap_start is not None:
                final_epoch = min(notes_cpap_start,field_cpap_start) - 60
            assert final_epoch > 20, f"Cpap machine turns on during epoch {final_epoch + 1}!"
        
        if end_time < final_epoch*EPOCH_LEN:
                final_epoch = len(signal_dict[self.channel_list[0]])//EPOCH_LEN
            
        end_time = final_epoch*EPOCH_LEN
        epochs = epochs[0:epochs.index(final_epoch)+1]
        stages = stages[0:epochs.index(final_epoch)+1]
        stage_dict = {epoch: stages[i] for i, epoch in enumerate(epochs)}
        self.stage_dict[ID] = stage_dict
        
        # Prepare for for making apnea channels
        X = np.zeros((end_time*self.fs,self.n_channels))
        print(f'Shape of X: {X.shape}')
        self.apnea_hypoapnea_targets[ID] = convert_to_sequence(time_list = self._extract_apnea_hypopnea_events(xdf),
                                                                         end_time = end_time, sampling_rate = self.fs)
                                                             
        for i,c in enumerate(self.channel_list):
            sig = signal_dict[c]
            print(f'\tProcessing channel: {c}')
            if c in ['Abdomen','Chest','P-Flo','Airflow']:
                # low-pass filter at 3 Hz
                sig = sig[:end_time]
                sig = lowpass(sig.ravel(),cutoff = 3, fs = 200)
                sig = resample(sig,200,10)
                if self.baseline:
                    sig = baseline(data=sig,
                                   labels = self.apnea_hypoapnea_targets[ID],
                                   baseline_type = 'min_max',
                                   cut_offs = (0.05,0.95),
                                   sampling_rate=10,
                                   baseline_length=120,
                                   step_size=10)
                elif self.normalize:
                    print(f'\tshape of sig: {sig.shape}')
                    transformer = Normalizer().fit(sig.T.reshape(-1, 1))
                    sig =  transformer.transform(sig.T.reshape(-1, 1)).T
                X[:,i] = sig
                
            elif c == 'SpO2':
                sig = sig[:end_time]
                sr_in, sr_out = sig.shape[-1], 10
                sig = resample(sig.ravel(), sr_in, sr_out, axis=-1)
                sig = sig*0.3663*0.001
                sig = np.clip(a=sig, a_min = 0, a_max = None)
                assert sig.shape == X[:,i].shape, f"Shape mismatch! Shape of signal: {sig.shape} and shape of slot: {X[:,i].shape}"
                if self.baseline:
                    X[:,i] = baseline(data=sig,
                                     labels = self.apnea_hypoapnea_targets[ID],
                                     baseline_type = 'quantile',
                                     sampling_rate=10,
                                     quantile=0.95,
                                     baseline_length=120,
                                     step_size=10)
                elif self.normalize:
                    transformer = Normalizer().fit(sig.T.reshape(-1, 1))
                    sig =  transformer.transform(sig.T.reshape(-1, 1)).T
                X[:,i] = sig
                    
            elif c == 'ECG':
                if self.hr == True:
                    ecg_sampling_rate = sig.shape[-1]
                    assert ecg_sampling_rate == 200, f"ECG samling rate is {ecg_sampling_rate}, not 200Hz!"
                    try:
                        # Sometimes too few heartbeats to calculate rate
                        ecg_data = ecg(signal = sig.ravel(), sampling_rate = ecg_sampling_rate, show = False)
                        mf = MedianFilter(filter_lower = self.ecg_filter_lower, filter_upper = self.ecg_filter_upper,
                                          upper_bound = self.ecg_upper_bound, lower_bound = self.ecg_lower_bound, 
                                          output_sampling_rate = self.fs)
                        hr_ts = mf.generate_hr_time_series(ecg_data = ecg_data, end_time = end_time)
                        X[:,i] = hr_ts[0:self.fs*end_time]
                    except Exception as e:
                        print("Error in calculating HR!!", e)
                        continue
                else:
                    sig = sig[:end_time]
                    sr_in, sr_out = sig.shape[-1], 10
                    sig = resample(sig.ravel(), sr_in, sr_out, axis=-1)
                    X[:,i] = sig

                
        # write out a file for each id
        file_name = f"{ID}"
        np.save(self.output_data_path.joinpath(file_name),X)

    def _extract_apnea_hypopnea_events(self, xdf):
        """
        Extract all apnea and hypopnea events from a study with respect to a subset of epochs
        """
        apnea_events = xdf.events["Dennis"].get("Apneas") or []
        hypopnea_events = xdf.events["Dennis"].get("Hypopneas") or []
        apnea_hypopnea_events = []
        for i, event in enumerate(apnea_events + hypopnea_events):
            t, d = event["Time"], event["Duration"]
            #print(f'type of d: {type(d)}')
            event_type = 'A' if i < len(apnea_events) else 'H'
            tt = (self.format_datetime(t) - xdf.start_time).total_seconds()
            apnea_hypopnea_events.append((round(tt, 6), round(tt + float(d), 6), event_type))
        return apnea_hypopnea_events


    @staticmethod
    def format_datetime(s):
        """
        Convert a date string of the format "%Y-%m-%dT%H:%M:%S.%f" to a datetime object
        """
        # The milliseconds can be an issue so need to be truncated in cases
        pre, suff = s.split(".")
        suff = suff[:6]
        sf = ".".join([pre, suff])
        return datetime.strptime(sf, "%Y-%m-%dT%H:%M:%S.%f")

    
    def process_data(self):
        n_tot = len(self.files)
        for i, f in enumerate(self.files):
            if self.ID_list is not None:
                if f.name not in self.ID_list:
                    continue
            print(f"{i + 1:3d} / {n_tot} {f.stem}")
            try:
                self._process_helper(f)

            except Exception as e:
                print("ERROR", e)
                continue
        
            
        # write out metadata files to output data dir
        with self.output_data_path.joinpath("stage_dict.p").open("wb") as fd:
            pickle.dump(self.stage_dict, fd)
                    
        with self.output_data_path.joinpath('apnea_hypopnea_targets.p').open('wb') as fah:
            pickle.dump(self.apnea_hypoapnea_targets,fah)


if __name__ == "__main__":
    import sys
    #print('here!')
    sys.path.append('/Users/danielyaeger/Documents/Modules')
    sys.path.append('/Users/danielyaeger/Documents/Modules/sleep-research-ml/src')
    ID_list = ['XAXVDJYND80Q4Q0',
               'XAXVDJYND82F3BO',
               'XAXVDJYND83LQ1F',
               'XAXVDJYND82Q807',
               'XAXVDJYND7ZN951',
               'XAXVDJYND7WLVFB',
               'XAXVDJYND7Q6JTK',
               'XAXVDJYND7ZUIM2']
    #print(sys.path)
    dg = DataProcessorApnea(ID_list = ID_list)
    dg.process_data()
    #create_data_partition(file_path = '/Volumes/Elements/selected_p_files_npy')