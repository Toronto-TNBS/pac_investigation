'''
Created on Apr 7, 2021

@author: voodoocode
'''

import numpy as np
import scipy.signal
import finn.filters.frequency as ff

def identify_peaks(in_data, fs, filter_f_min = 300, filter_f_max = None, spread = 1.5, peak_thresh = 1.1, polarity = "negative", 
                   buffer_area = None):
    
    if (buffer_area == "auto"):
        buffer_area = int(fs/300)

    binarized_data = np.ones(in_data.shape) * -1
    in_data_mod = ff.fir(in_data, filter_f_min, filter_f_max, 1, fs)
    if (polarity == "negative"):
        in_data_mod[in_data_mod > 0] = 0
    elif(polarity == "positive"):
        in_data_mod[in_data_mod < 0] = in_data_mod
    else:
        raise AssertionError("Polarity must be either 'positive' or 'negative'")
    
    if (buffer_area is not None):
        (peaks, _) = scipy.signal.find_peaks(np.abs(in_data_mod), height = peak_thresh, distance = buffer_area)
    else:
        (peaks, _) = scipy.signal.find_peaks(np.abs(in_data_mod), height = peak_thresh)
    
    if (len(peaks) == 0):
        return np.ones(in_data.shape) * -1
    
    samples_between_instantaneous_spikes = len(in_data_mod)/len(peaks)
    max_dist_thresh = samples_between_instantaneous_spikes * (1/spread)
        
    last_peak = peaks[0]
    burst_start = peaks[0]
    for peak in peaks[1:]:
        if ((peak - last_peak) > max_dist_thresh):
            if (last_peak != burst_start):
                binarized_data[burst_start:np.min(((last_peak + 1), len(binarized_data)))] = 1
     
                if (buffer_area is not None):
                    binarized_data[np.max((burst_start - buffer_area, 0)):burst_start] = 0
                    binarized_data[(last_peak + 1):np.min(((last_peak + 1 + buffer_area), len(binarized_data)))] = 0
            
            #===================================================================
            # binarized_data[burst_start:(last_peak + 1)] = 1
            #===================================================================
            last_peak = peak
            burst_start = peak
        
        last_peak = peak
        
    return binarized_data









