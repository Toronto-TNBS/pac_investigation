'''
Created on Apr 7, 2021

@author: voodoocode
'''

import numpy as np
import scipy.signal
import finn.filters.frequency as ff

def identify_peaks(in_data, fs, filter_f_min = 70, filter_f_max = None, spread = 1.5, peak_thresh = 1.1, polarity = "negative"):

    binarized_data = np.zeros(in_data.shape)
    in_data_mod = ff.fir(in_data, filter_f_min, filter_f_max, 1, fs)
    if (polarity == "negative"):
        in_data_mod[in_data_mod > 0] = 0
    elif(polarity == "positive"):
        in_data_mod[in_data_mod < 0] = in_data_mod
    else:
        raise AssertionError("Polarity must be either 'positive' or 'negative'")
        
    (peaks, _) = scipy.signal.find_peaks(np.abs(in_data_mod), height = peak_thresh)
    
    samples_between_instantaneous_spikes = len(in_data_mod)/len(peaks)
    max_dist_thresh = samples_between_instantaneous_spikes * (1/spread)
        
    last_peak = peaks[0]
    burst_start = peaks[0]
    for peak in peaks[1:]:
        if ((peak - last_peak) > max_dist_thresh):
            binarized_data[burst_start:last_peak] = 1#np.mean(data[burst_start:last_peak])
            last_peak = peak
            burst_start = peak
        
        last_peak = peak
        
    return binarized_data