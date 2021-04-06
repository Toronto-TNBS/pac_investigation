# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: Luka
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

import scipy.signal

import finn.filters.frequency as ff
import finn.cross_frequency_coupling.direct_modulation_index as dmi
import scipy.optimize

import pandas

import finn.misc.timed_pool as tp

import os.path

import finn.artifact_rejection.outlier_removal as aror

thread_cnt = 4

### 3050 has nice beta
### Feb 1 935 tremor cell stn

######################### PATIENT 1 - 2626 ############################################################################### DONE
######################### BETA_NEURON 1 
# BETA nice PAC [center 21]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1 TIGHT BURSTING
#9s
hdr01 = pickle.load(open("../../data/data_for_python/2626-s4-568-b.txt_conv_hdr.pkl", "rb"))
data01 = pickle.load(open("../../data/data_for_python/2626-s4-568-b.txt_conv_data.pkl", "rb"))

#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_data.pkl", "rb"))

filt_low = 16
filt_high = 25
smooth_factor = 0.01

######################### BETA_NEURON 2
# intermittent BETA... [center 31]
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1665-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1665-a.txt_conv_data.pkl", "rb"))
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### BETA_NEURON 3
# HUH-BETA [center 31] decent NOT ORIGINALLY CLASSIFIED AS BETA...
#12s
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1085.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1085.txt_conv_data.pkl", "rb"))
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### BETA_NEURON 4
# BETA... [31ish peak]
#20s
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-704.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-704.txt_conv_data.pkl", "rb"))
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA... inverse
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-557.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-557.txt_conv_data.pkl", "rb"))
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# NO PAC AT ALL
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1340.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1340.txt_conv_data.pkl", "rb"))
#filt_low = 26
#filt_high = 36
#smooth_factor = 0.01



######################### PATIENT 2 - 2767 ############################################################################### DONE
######################### BETA_NEURON 1 - PUB FIG!
# nice PAC [center 24]
#hdr01 = pickle.load(open("../../data/data_for_python/2767-S1-1330-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-S1-1330-b.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA
#hdr01 = pickle.load(open("../../data/data_for_python/2767-s1-890-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-s1-890-a.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA
#hdr01 = pickle.load(open("../../data/data_for_python/2767-s1-1050-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-s1-1050-a.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01



######################### PATIENT 3 - 2829 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [center 14]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-1074.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-1074.txt_conv_data.pkl", "rb"))
#filt_low = 10
#filt_high = 18
#smooth_factor = 0.01

######################### BETA_NEURON 2
# nice PAC - PUB FIGGGG!!!! [centers 16]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~5
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-1240-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-1240-a.txt_conv_data.pkl", "rb"))
#filt_low = 10
#filt_high = 20
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s2-416-no-beta.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s2-416-no-beta.txt_conv_data.pkl", "rb"))
#filt_low = 25
#filt_high = 35
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_data.pkl", "rb"))
#filt_low = 10
#filt_high = 20
#smooth_factor = 0.01



######################### PATIENT 4 - 2884 ############################################################################### DONE
######################### BETA_NEURON 1...?? CONFUSING NEURON
# not bad... [peak at 16] 180 deg phase shift
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-393.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-393.txt_conv_data.pkl", "rb"))
## same neuron
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-515.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-515.txt_conv_data.pkl", "rb"))
#filt_low = 11
#filt_high = 21
#smooth_factor = 0.01

######################### BETA_NEURON 2
# nice PAC [low center... 11?]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~6
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-615-c.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-615-c.txt_conv_data.pkl", "rb"))
#filt_low = 8
#filt_high = 14
#smooth_factor = 0.01

######################### BETA_NEURON 3
# nice PAC [peak at 12]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~7
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-740.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-740.txt_conv_data.pkl", "rb"))
#filt_low = 9
#filt_high = 15
#smooth_factor = 0.01

######################### BETA_NEURON 4
# dece PAC [16 center]
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-755.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-755.txt_conv_data.pkl", "rb"))
#filt_low = 11
#filt_high = 21
#smooth_factor = 0.01

######################### BETA_NEURON 5
# dece PAC [16 center]
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-586.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-586.txt_conv_data.pkl", "rb"))
#filt_low = 11
#filt_high = 21
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta ... no peaks
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-945.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-945.txt_conv_data.pkl", "rb"))
#filt_low = 11
#filt_high = 21
#smooth_factor = 0.01


######################### PATIENT 5 - 2903 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [peak at 28] (PUB!!!!)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~8
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-650.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-650.txt_conv_data.pkl", "rb"))
#filt_low = 24
#filt_high = 32
#smooth_factor = 0.01

######################### BETA_NEURON 2
# nice PAC [peak at 28] (PUB!!!!)
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-775-long.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-775-long.txt_conv_data.pkl", "rb"))
#filt_low = 25
#filt_high = 34
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-987.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-987.txt_conv_data.pkl", "rb"))
#filt_low = 25
#filt_high = 34
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s1-1078.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s1-1078.txt_conv_data.pkl", "rb"))
#filt_low = 24
#filt_high = 32
#smooth_factor = 0.01



######################### PATIENT 6 - 2623 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [24 peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~9
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-508.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-508.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-385.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-385.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-666.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-666.txt_conv_data.pkl", "rb"))
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 3
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s4-280.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s4-280.txt_conv_data.pkl", "rb"))
#filt_low = 21
#filt_high = 31
#smooth_factor = 0.01

######################### NON-BETA_NEURON 4
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s1-130.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s1-130.txt_conv_data.pkl", "rb"))
#filt_low = 21
#filt_high = 31
#smooth_factor = 0.01



######################### PATIENT ... - 3137 - actual ptn was 678 ###############################################################################
######################### BETA_NEURON 1
# nice PAC [24 peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~9
#hdr01 = pickle.load(open("../../data/data_for_python/3137-s-3.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/3137-s-3.txt_conv_data.pkl", "rb"))
#filt_low = 9
#filt_high = 18
#smooth_factor = 0.01

######################### PATIENT ... - 2810 ############################################################################### PRETTY CRAP
######################### BETA_NEURON 1
# fine PAC [NOT GREAT.. weak 22hz peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3
#hdr01 = pickle.load(open("../../data/data_for_python/2810-s2-950.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2810-s2-950.txt_conv_data.pkl", "rb"))
#filt_low = 18
#filt_high = 28
#smooth_factor = 0.01
# fine PAC [NOT GREAT.. weak 22hz peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3
#hdr01 = pickle.load(open("../../data/data_for_python/2810-s2-950-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2810-s2-950-b.txt_conv_data.pkl", "rb"))
#filt_low = 8
#filt_high = 18
#smooth_factor = 0.01


#To Do
#Filter bursts only
#Filter non-bursts only 


def smooth_data(in_data, fs):
    winWidth = int(0.01*fs*2)
    smoothed_data01 = np.pad(in_data, winWidth, 'constant', constant_values = [0]) 
    smoothed_data01 = np.asarray(pandas.Series(np.abs(smoothed_data01)).rolling(center = True, window = int(smooth_factor*fs)).mean())
    return smoothed_data01[winWidth:-winWidth]

def preprocess_data(in_data, fs):
    in_data = np.copy(in_data)
    
    if (burst_filter == "spike_transform"):
        out_data = transform_spike_data(in_data, fs)
    elif(burst_filter == "hilbert"):
        out_data = np.abs(scipy.signal.hilbert(in_data))
    elif(burst_filter == "basic"):
        out_data = np.copy(in_data)
    else:
        raise NotImplementedError
    return out_data

def plot_hf_lf_components(data, fs_data01, f_min = 5, f_max = 45, visualize = True):
    # filtering
    hpf_data01 = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
    bpf_data01 = ff.fir(np.asarray(data), filt_low, filt_high, 0.1, fs_data01)

    # spike signal smooth-transform
    smoothed_data01 = preprocess_data(hpf_data01, fs_data01)
        
    (_, smoothed_hpf_psd) = scipy.signal.welch(smoothed_data01, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
    (bins, bpf_psd) = scipy.signal.welch(bpf_data01, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
    
    start_bin = np.argmin(np.abs(bins - f_min))
    end_bin = np.argmin(np.abs(bins - f_max))
    
    if (visualize):
        (_, axes) = plt.subplots(2, 3)
        t_data01 = np.arange(1, len(hpf_data01)+1, 1)/fs_data01
        axes[0, 0].plot(t_data01, hpf_data01)
        axes[0, 0].plot(t_data01, smoothed_data01)
        axes[1, 0].plot(t_data01, bpf_data01)
        
        axes[0, 1].semilogy(bins[start_bin:end_bin], smoothed_hpf_psd[start_bin:end_bin])
        axes[1, 1].semilogy(bins[start_bin:end_bin], bpf_psd[start_bin:end_bin])
        
        plt.figure()
        plt.plot(t_data01, hpf_data01)
        plt.plot(t_data01, smoothed_data01)

def transform_spike_data(data, fs):
    
    
    #Add: remove hf spikes (single spikes) for better results?
    
    data*=-1
    data[np.abs(data) < 1] = 0
    data*= 1000
    
    win_sz = 160
    data = np.pad(data, win_sz)
    data = pandas.Series(data).rolling(center = True, window = win_sz).mean()
    data = data[win_sz:-win_sz]
    data = np.asarray(data)
    
    return data
    
    
    
    binarized_data = np.zeros(data.shape)
        
    (peaks, _) = scipy.signal.find_peaks(np.abs(data), height = 1.1)
    
    samples_between_instantaneous_spikes = len(data)/len(peaks)
    max_dist_thresh = samples_between_instantaneous_spikes * 2 / 3
        
    last_peak = peaks[0]
    burst_start = peaks[0]
    for peak in peaks[1:]:
        if ((peak - last_peak) > max_dist_thresh):
            binarized_data[burst_start:last_peak] = 1#np.mean(data[burst_start:last_peak])
            last_peak = peak
            burst_start = peak
        
        last_peak = peak
        
    return binarized_data
        
    
        
    #===========================================================================
    # (peaks, _) = scipy.signal.find_peaks(binarized_data)
    # (vallies, _) = scipy.signal.find_peaks(binarized_data * -1)
    # 
    # if (vallies[0] > peaks[0]):
    #     vallies = np.concatenate(([0], vallies))
    # 
    # peaks_vallies = np.empty((peaks.size + vallies.size,), dtype=peaks.dtype)
    # peaks_vallies[0::2] = vallies
    # peaks_vallies[1::2] = peaks
    # 
    # interp_data = np.zeros(binarized_data.shape)
    # for mrk_idx in range(len(peaks_vallies) - 1):
    #     start = peaks_vallies[mrk_idx]
    #     end = peaks_vallies[mrk_idx + 1]
    #     
    #     weight_vector = 1 / (1 + np.exp(-np.arange(-6, 6, 12/(end-start))))
    #     for smpl_idx in range(0, end - start, 1):
    #         interp_data[smpl_idx + start] = binarized_data[start] * (1 - weight_vector[smpl_idx]) + binarized_data[end] * weight_vector[smpl_idx]
    # 
    # plt.figure()
    # plt.plot(interp_data)
    # plt.show(block = True)
    #===========================================================================
    
    
    (peaks, _) = scipy.signal.find_peaks(binarized_data)
    
    vallies = list()
    for data_pt_idx in np.arange(1, len(binarized_data) - 1, 1):
        if (binarized_data[data_pt_idx] == 0 and (binarized_data[data_pt_idx - 1] == 1 or binarized_data[data_pt_idx+ 1] == 1)):
            vallies.append(data_pt_idx)
    vallies = np.asarray(vallies)
    
    if (vallies[0] > peaks[0]):
        vallies = np.concatenate(([0], vallies))
    
    peaks_vallies = np.empty((peaks.size + vallies.size,), dtype=peaks.dtype)
    peaks_vallies[0::3] = vallies[::2]
    peaks_vallies[1::3] = peaks
    peaks_vallies[2::3] = vallies[1::2]
    
    interp_data = np.zeros(binarized_data.shape)
    for mrk_idx in np.arange(0, len(peaks_vallies), 3):
        start = peaks_vallies[mrk_idx]
        middle = peaks_vallies[mrk_idx + 1]
        end = peaks_vallies[mrk_idx + 2]
                
        weight_vector_0 = 1 / (1 + np.exp(-np.arange(-6, 6, 12/(middle-start))))
        weight_vector_1 = 1 / (1 + np.exp(-np.arange(-6, 6, 12/(end-middle))))
        for smpl_idx in range(0, middle - start, 1):
            interp_data[smpl_idx + start] = binarized_data[start] * (1 - weight_vector_0[smpl_idx]) + binarized_data[middle] * weight_vector_0[smpl_idx]
        for smpl_idx in range(0, end - middle, 1):
            interp_data[smpl_idx + middle] = binarized_data[middle] * (1 - weight_vector_1[smpl_idx]) + binarized_data[end] * weight_vector_1[smpl_idx]
    
    return interp_data

def calculate_dmi(data, fs_data01, visualize = True):
    high_freq_component = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
    low_freq_component = ff.fir(np.asarray(data), filt_low, filt_high, 0.1, fs_data01)
    
    smoothed_high_freq_component = preprocess_data(high_freq_component, fs_data01)
    
    (score, best_fit, amp_signal) = dmi.run(low_freq_component, high_freq_component, 10, 1)
    (smoothed_score, smoothed_best_fit, smoothed_amp_signal) = dmi.run(low_freq_component, smoothed_high_freq_component, 10, 1)

    if (visualize):
        (fig, axes) = plt.subplots(2, 1)
        axes[0].set_title("original data, score: %.2f" % (score,))
        axes[0].plot(best_fit)
        axes[0].plot(amp_signal)
        
        axes[1].set_title("smoothed data, score: %.2f" % (smoothed_score,))
        axes[1].plot(smoothed_best_fit)
        axes[1].plot(smoothed_amp_signal)
        
        fig.set_tight_layout(tight = True)
     
    return (score, best_fit, amp_signal)

def calculate_spectograms_inner(data, window_width, fs, f_min, f_max, f_window_width, f_window_step_sz, start_idx, filter_step_width = 1):        
    print("Processing idx %i of %i" % (start_idx, len(data) - window_width))
    data = np.asarray(data)
       
    loc_hf_data = ff.fir(np.copy(data), 300, None, filter_step_width, fs)
    loc_lf_data = ff.fir(np.copy(data), filt_low, filt_high, filter_step_width, fs)
    
    loc_hf_data = preprocess_data(loc_hf_data, fs)

    (bins, loc_hf_psd) = scipy.signal.welch(loc_hf_data, fs, window = "hann", nperseg = fs, noverlap = int(fs/2), nfft = fs, detrend = False, return_onesided = True)
    (_, loc_lf_psd) = scipy.signal.welch(loc_lf_data, fs, window = "hann", nperseg = fs, noverlap = int(fs/2), nfft = fs, detrend = False, return_onesided = True)
    min_f_bin_idx = np.argmin(np.abs(bins - f_min))
    max_f_bin_idx = np.argmin(np.abs(bins - f_max))
    
    loc_dmi_scores = list()
    for f_idx in np.arange(f_min, f_max, f_window_step_sz):
        loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)
        (loc_dmi_score, _, _) = dmi.run(loc_dmi_lf_data, loc_hf_data, 10, 1)
        loc_dmi_scores.append(loc_dmi_score)
            
    return (loc_hf_psd[min_f_bin_idx:max_f_bin_idx], loc_lf_psd[min_f_bin_idx:max_f_bin_idx], loc_dmi_scores)

def calculate_spectograms(data, fs, visualize = True):
    window_width = fs
    window_step_sz = int(fs/2)
    
    f_min = 5
    f_max = 45
    f_window_width = 2
    f_window_step_sz = 1
    
    #===========================================================================
    # for start_idx in np.arange(0, len(data) - window_width, window_step_sz):
    #     calculate_spectograms_inner(start_idx, data, window_width, fs, f_min, f_max, f_window_width, f_window_step_sz)
    #===========================================================================
    
    #===========================================================================
    # if (os.path.exists("test.npy")):
    #     tmp = np.load("test.npy", allow_pickle = True)
    # else:
    #     tmp = np.asarray(tp.run(8, calculate_spectograms_inner,
    #                             [(data[start_idx:(start_idx + window_width)], window_width, fs, f_min, f_max, f_window_width, f_window_step_sz, start_idx) for start_idx in np.arange(0, len(data) - window_width, window_step_sz)],
    #                             max_time = None, delete_data = True))   
    #     np.save("test.npy", tmp, allow_pickle = True)
    #===========================================================================
        
    tmp = np.asarray(tp.run(thread_cnt, calculate_spectograms_inner,
                            [(data[start_idx:(start_idx + window_width)], window_width, fs, f_min, f_max, f_window_width, f_window_step_sz, start_idx) for start_idx in np.arange(0, len(data) - window_width, window_step_sz)],
                             max_time = None, delete_data = True))
        
    hf_psd = np.asarray(list(tmp[:, 0]))
    lf_psd = np.asarray(list(tmp[:, 1]))
    dmi_scores = np.asarray(list(tmp[:, 2]))
        
    if (visualize):
    
        hf_psd = np.flip(hf_psd, axis = 1)
        lf_psd = np.flip(lf_psd, axis = 1)
        dmi_scores = np.flip(dmi_scores, axis = 1)
        
        hf_psd = np.transpose(hf_psd)
        lf_psd = np.transpose(lf_psd)
        dmi_scores = np.transpose(dmi_scores)
        
        #=======================================================================
        # lf_psd = np.log10(lf_psd)
        # hf_psd = np.log10(hf_psd)
        #=======================================================================
        
        (fig, axes) = plt.subplots(3, 1)
        axes[0].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        axes[1].imshow(hf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        #=======================================================================
        # axes[0].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -4.5)
        # #axes[1].imshow(hf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -3.8)
        # axes[1].imshow(hf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = 0.75)
        #=======================================================================
        axes[2].imshow(dmi_scores, vmin = 0.75, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        
        axes[0].set_title("low frequency psd")
        axes[1].set_title("high frequency psd")
        axes[2].set_title("PAC: low freq. - high freq")
        
        for ax in axes:
            ax.set_yticks([0, 4, 9, 14, 19, 24, 29, 34, 39])
            ax.set_yticklabels(([5, 10, 15, 20, 25, 30, 35, 40, 45][::-1]))
        
        fig.set_tight_layout(tight = True)
        
def main(all_data, hdr_info, overwrite = False):

    
    fs_data01 = int(hdr01[20]['fs'])
    print(fs_data01)
    
    plot_hf_lf_components(data01[20], fs_data01)
    calculate_dmi(data01[20], fs_data01)
    calculate_spectograms(data01[20], fs_data01)
    
    plt.show(block = True)
    print("Terminated successfully")

#burst_filter = "hilbert"
burst_filter = "spike_transform"
#burst_filter = "basic"

main(data01, hdr01, False)


