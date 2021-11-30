# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: Luka & VoodooCode
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.signal

import finn.filters.frequency as ff
import finn.cfc.pac as pac

import finn.misc.timed_pool as tp

import methods.detection.bursts
import methods.data_io.ods

import os
import scipy.stats
import pandas

import os.path

import pandas

thread_cnt = 11
#thread_cnt = 1
#image_format = ".png"
image_format = ".svg"

def preprocess_data(in_data, fs, peak_spread, peak_thresh, mode = "gaussian"):
    in_data = np.copy(in_data)
    binarized_data = methods.detection.bursts.identify_peaks(in_data, fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
    
    burst_data = transform_burst(in_data, binarized_data, mode)
    non_burst_data = transform_non_burst(in_data, binarized_data, mode)
    
    return (burst_data, non_burst_data)

def transform_burst(in_data, binarized_data, mode):
    if (mode == "gaussian"):
        out_data = np.random.normal(loc = 0, scale = np.mean(np.abs(in_data[np.argwhere(binarized_data == -1).squeeze()])), size = in_data.shape[0])
    elif(mode == "zero"):
        out_data = np.zeros(in_data.shape)

    out_data[np.argwhere(binarized_data == 1).squeeze()] = in_data[np.argwhere(binarized_data == 1).squeeze()]
    
    out_data = np.abs(scipy.signal.hilbert(out_data))
    
    return out_data

def transform_non_burst(in_data, binarized_data, mode):
    if (mode == "gaussian"):
        out_data = np.random.normal(loc = 0, scale = np.mean(np.abs(in_data[np.argwhere(binarized_data == -1).squeeze()])), size = in_data.shape[0])
    elif(mode == "zero"):
        out_data = np.zeros(in_data.shape)
        
    out_data[np.argwhere(binarized_data == -1).squeeze()] = in_data[np.argwhere(binarized_data == -1).squeeze()]
    
    out_data = np.abs(scipy.signal.hilbert(out_data))
    
    return out_data

def plot_hf_lf_components(data, fs_data01, lf_min_f, lf_max_f, hf_min_f, hf_max_f, f_min = 2, f_max = 45, peak_spread = 1.5, peak_thresh = 1.1, 
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (overwrite or os.path.exists(outpath + "data/1/" + file + ".pkl") == False):
        # filtering
        lpf_data01 = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        bpf_data01 = ff.fir(np.asarray(data), lf_min_f, lf_max_f, 0.1, fs_data01)
        hpf_data01 = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
        
        # spike signal smooth-transform
        (burst_hf_data, non_burst_hf_data) = preprocess_data(hpf_data01, fs_data01, peak_spread, peak_thresh, mode = "zero")
            
        (bins, lpf_psd) = scipy.signal.welch(lpf_data01, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
        (bins, bpf_psd) = scipy.signal.welch(bpf_data01, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
        (_, raw_hf_data_psd) = scipy.signal.welch(np.abs(scipy.signal.hilbert(hpf_data01)), fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
        (_, burst_hf_data_psd) = scipy.signal.welch(burst_hf_data, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
        (_, non_burst_hf_data_psd) = scipy.signal.welch(non_burst_hf_data, fs_data01, window = "hanning", nperseg = fs_data01, noverlap = int(fs_data01/2), nfft = fs_data01, detrend = False, return_onesided = True)
        
        start_bin = np.argmin(np.abs(bins - f_min))
        end_bin = np.argmin(np.abs(bins - f_max))
        
        t_data01 = np.arange(1, len(hpf_data01)+1, 1)/fs_data01
        
        pickle.dump([t_data01, lpf_data01, bpf_data01,
                     hpf_data01, burst_hf_data, non_burst_hf_data, 
                     bins, lpf_psd, bpf_psd,
                     raw_hf_data_psd, burst_hf_data_psd, non_burst_hf_data_psd],
                     open(outpath + "data/1/" + file + ".pkl", "wb"))
     
    else:
        tmp = pickle.load(open(outpath + "data/1/" + file + ".pkl", "rb"))
        t_data01 = tmp[0]; lpf_data01 = tmp[1]; bpf_data01 = tmp[2];
        hpf_data01 = tmp[3]; burst_hf_data = tmp[4]; non_burst_hf_data = tmp[5];
        bins = tmp[6]; lpf_psd = tmp[7]; bpf_psd = tmp[8];
        raw_hf_data_psd = tmp[9]; burst_hf_data_psd = tmp[10]; non_burst_hf_data_psd = tmp[11];
     
        start_bin = np.argmin(np.abs(bins - f_min))
        end_bin = np.argmin(np.abs(bins - f_max))
     
    ref_f_min = 2
    ref_f_max = 150
    ref_f_min = 2
    ref_f_max = 80
    ref_f_width = 5
    
    #lf_beta_strength = (np.sum(lpf_psd[lf_min_f:filt_high])/(filt_high-lf_min_f)) / (np.sum(lpf_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    #default_beta_strength = (np.sum(raw_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(raw_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    #burst_beta_strength = (np.sum(burst_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(burst_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    #non_burst_beta_strength = (np.sum(non_burst_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(non_burst_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    
    #lf_beta_strength = np.min((np.average(lpf_psd[filt_low:filt_high]) / np.average(lpf_psd[(filt_low-ref_f_width):filt_low]), np.average(lpf_psd[filt_low:filt_high]) / np.average(lpf_psd[filt_high:(filt_high+ref_f_width)])))
    lf_beta_strength = np.average(lpf_psd[lf_min_f:lf_max_f]) / np.average([np.average(lpf_psd[(lf_min_f-ref_f_width):lf_min_f]), np.average(lpf_psd[lf_max_f:(lf_max_f+ref_f_width)])])
    hf_beta_strength = np.average(raw_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(raw_hf_data_psd[(hf_min_f-ref_f_width):hf_min_f]), np.average(raw_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    burst_beta_strength = np.average(burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(burst_hf_data_psd[(hf_min_f-ref_f_width):hf_min_f]), np.average(burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    non_burst_beta_strength = np.average(non_burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(non_burst_hf_data_psd[(hf_min_f-ref_f_width):hf_min_f]), np.average(non_burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    
    lf_beta_strength = np.log(lf_beta_strength)
    hf_beta_strength = np.log(hf_beta_strength)
    burst_beta_strength = np.log(burst_beta_strength)
    non_burst_beta_strength = np.log(non_burst_beta_strength)
    
    #===========================================================================
    # lf_beta_strength = (np.sum(lpf_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(lpf_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    # default_beta_strength = (np.sum(raw_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(raw_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    # burst_beta_strength = (np.sum(burst_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(raw_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    # non_burst_beta_strength = (np.sum(non_burst_hf_data_psd[filt_low:filt_high])/(filt_high-filt_low)) / (np.sum(raw_hf_data_psd[ref_f_min:ref_f_max]) / (ref_f_max - ref_f_min))
    #===========================================================================
    
    #===========================================================================
    # lf_beta_strength = np.sum(lpf_psd[filt_low:filt_high])/np.sum(lpf_psd[ref_f_min:ref_f_max])
    # default_beta_strength = np.sum(raw_hf_data_psd[filt_low:filt_high])/np.sum(raw_hf_data_psd[ref_f_min:ref_f_max])
    # burst_beta_strength = np.sum(burst_hf_data_psd[filt_low:filt_high])/np.sum(burst_hf_data_psd[ref_f_min:ref_f_max])
    # non_burst_beta_strength = np.sum(non_burst_hf_data_psd[filt_low:filt_high])/np.sum(non_burst_hf_data_psd[ref_f_min:ref_f_max])
    #===========================================================================
    
    if (visualize):
        (fig, axes) = plt.subplots(3, 2)
        axes[0, 0].plot(t_data01[0:int(fs_data01/2)], lpf_data01[0:int(fs_data01/2)], color = "blue")
        axes[0, 0].plot(t_data01[0:int(fs_data01/2)], bpf_data01[0:int(fs_data01/2)], color = "orange")
        axes[1, 0].plot(t_data01[0:int(fs_data01/2)], hpf_data01[0:int(fs_data01/2)], color = "blue")
        axes[1, 0].plot(t_data01[0:int(fs_data01/2)], burst_hf_data[0:int(fs_data01/2)], color = "orange")
        axes[2, 0].plot(t_data01[0:int(fs_data01/2)], hpf_data01[0:int(fs_data01/2)], color = "blue")
        axes[2, 0].plot(t_data01[0:int(fs_data01/2)], non_burst_hf_data[0:int(fs_data01/2)], color = "orange")
        
        vmax = np.max([np.max(burst_hf_data_psd[start_bin:end_bin]), np.max(non_burst_hf_data_psd[start_bin:end_bin])]) 
        axes[0, 1].plot(bins[start_bin:end_bin], lpf_psd[start_bin:end_bin], color = "black")
        axes[0, 1].plot(bins[start_bin:end_bin], bpf_psd[start_bin:end_bin], color = "blue")
        axes[1, 1].plot(bins[start_bin:end_bin], raw_hf_data_psd[start_bin:end_bin], color = "black")
        axes[1, 1].plot(bins[start_bin:end_bin], burst_hf_data_psd[start_bin:end_bin], color = "blue")
        axes[2, 1].plot(bins[start_bin:end_bin], raw_hf_data_psd[start_bin:end_bin], color = "black")
        axes[2, 1].plot(bins[start_bin:end_bin], non_burst_hf_data_psd[start_bin:end_bin], color = "blue")
        
        axes[0, 0].set_title("default")
        axes[1, 0].set_title("burst")
        axes[2, 0].set_title("non burst")
        axes[0, 1].set_title("default %1.2f" % (lf_beta_strength,))
        axes[1, 1].set_title("burst %1.2f of %1.2f" % (burst_beta_strength, hf_beta_strength))
        axes[2, 1].set_title("non burst %1.2f of %1.2f" % (non_burst_beta_strength, hf_beta_strength))
        
        axes[1, 1].set_yticks([0, vmax])
        axes[2, 1].set_yticks([0, vmax])
        
        axes[0, 1].set_xticks(np.arange(2, 45, 2))
        axes[1, 1].set_xticks(np.arange(2, 45, 2))
        axes[2, 1].set_xticks(np.arange(2, 45, 2))
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticks(), rotation = 45)
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticks(), rotation = 45)
        axes[2, 1].set_xticklabels(axes[2, 1].get_xticks(), rotation = 45)
        
        fig.set_tight_layout(tight = True)
        fig.savefig(outpath + "img/1/" + file + image_format)
        plt.close()
            
    return (lf_beta_strength, hf_beta_strength, burst_beta_strength, non_burst_beta_strength)

def calculate_pac(data, fs_data01, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    if (overwrite or os.path.exists(outpath + "data/2/" + file + ".pkl") == False):
    
        high_freq_component = ff.fir(np.asarray(data), 300, None, 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), lf_f_min, lf_f_max, 0.1, fs_data01)
        low_freq_component = ff.fir(np.asarray(data), hf_f_min, hf_f_max, 0.1, fs_data01)
        
        #low_freq_component = ff.fir(np.asarray(data), np.min([lf_f_min, hf_f_min]), np.max([lf_f_max, hf_f_max]), 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), 12, 32, 0.1, fs_data01)
        
        (burst_hf_data, non_burst_hf_data) = preprocess_data(high_freq_component, fs_data01, peak_spread, peak_thresh)
        
        ### START
        binarized_data = methods.detection.bursts.identify_peaks(np.copy(high_freq_component), fs_data01, 300, None, peak_spread, peak_thresh, "negative", "auto")
        random_data = np.random.normal(loc = 0, scale = np.mean(np.abs(high_freq_component[np.argwhere(binarized_data == -1).squeeze()])), size = high_freq_component.shape[0])
        non_burst_hf_data2 = np.copy(random_data)
        non_burst_hf_data2[np.argwhere(binarized_data == -1).squeeze()] = high_freq_component[np.argwhere(binarized_data == -1).squeeze()]
        
        pre_peak_data = ff.fir(np.copy(high_freq_component), 300, None, 1, fs_data01)
        pre_peak_data[pre_peak_data > 0] = 0
        (peaks, _) = scipy.signal.find_peaks(np.abs(pre_peak_data), height = peak_thresh, distance = int(fs_data01/300))
        
        non_burst_hf_data3 = np.copy(random_data)
        for peak in peaks:
            if (np.abs(non_burst_hf_data2[peak]) < peak_thresh):
                continue
            
            start = int(peak - 10)
            end = int(peak + 20)
            
            non_burst_hf_data2[start:end] = random_data[start:end]
            non_burst_hf_data3[start:end] = high_freq_component[start:end]
        
        non_burst_hf_data2 = np.abs(scipy.signal.hilbert(non_burst_hf_data2))
        non_burst_hf_data3 = np.abs(scipy.signal.hilbert(non_burst_hf_data3))
        
        ### END
        
        (score_default, best_fit_default, amp_signal_default) = pac.run_dmi(low_freq_component, high_freq_component, 10, 1)
        (score_burst, best_fit_burst, amp_signal_burst) = pac.run_dmi(low_freq_component, burst_hf_data, 10, 1)
        (score_non_burst, best_fit_non_burst, amp_signal_non_burst) = pac.run_dmi(low_freq_component, non_burst_hf_data, 10, 1)
        (score_non_burst2, best_fit_non_burst2, amp_signal_non_burst2) = pac.run_dmi(low_freq_component, non_burst_hf_data2, 10, 1)
        (score_non_burst3, best_fit_non_burst3, amp_signal_non_burst3) = pac.run_dmi(low_freq_component, non_burst_hf_data3, 10, 1)
        
        pickle.dump([best_fit_default, amp_signal_default, score_default,
                     best_fit_burst, amp_signal_burst, score_burst,
                     best_fit_non_burst, amp_signal_non_burst, score_non_burst, 
                     best_fit_non_burst2, amp_signal_non_burst2, score_non_burst2, 
                     best_fit_non_burst3, amp_signal_non_burst3, score_non_burst3], open(outpath + "data/2/" + file + ".pkl", "wb"))
        
    else:
        
        tmp = pickle.load(open(outpath + "data/2/" + file + ".pkl", "rb"))
        best_fit_default = tmp[0]; amp_signal_default = tmp[1]; score_default = tmp[2];
        best_fit_burst = tmp[3]; amp_signal_burst = tmp[4]; score_burst = tmp[5];
        best_fit_non_burst = tmp[6]; amp_signal_non_burst = tmp[7]; score_non_burst = tmp[8];
    
    if (visualize):
        (fig, axes) = plt.subplots(3, 1)
        axes[0].set_title("default: original data, score: %.2f" % (score_default,))
        axes[0].plot(best_fit_default)
        axes[0].plot(amp_signal_default)
        
        axes[1].set_title("burst: smoothed data, score: %.2f" % (score_burst,))
        axes[1].plot(best_fit_burst)
        axes[1].plot(amp_signal_burst)
        
        axes[2].set_title("non burst: smoothed data, score: %.2f" % (score_non_burst,))
        axes[2].plot(best_fit_non_burst)
        axes[2].plot(amp_signal_non_burst)
        
        fig.set_tight_layout(tight = True)
        fig.savefig(outpath + "img/2/" + file + image_format)
        plt.close()
        
    return (score_default, score_burst, score_non_burst)

def default_beta_phase_pac(data, fs_data01, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    if (overwrite or os.path.exists(outpath + "data/2/" + file + ".pkl") == False):
    
        high_freq_component = ff.fir(np.asarray(data), 300, None, 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), lf_f_min, lf_f_max, 0.1, fs_data01)
        low_freq_component = ff.fir(np.asarray(data), 12, 32, 0.1, fs_data01)
        
        #low_freq_component = ff.fir(np.asarray(data), np.min([lf_f_min, hf_f_min]), np.max([lf_f_max, hf_f_max]), 0.1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), 12, 32, 0.1, fs_data01)
        
        (burst_hf_data, non_burst_hf_data) = preprocess_data(high_freq_component, fs_data01, peak_spread, peak_thresh)
        
        (score_default, best_fit_default, amp_signal_default) = pac.run_dmi(low_freq_component, high_freq_component, 10, 1)
        (score_burst, best_fit_burst, amp_signal_burst) = pac.run_dmi(low_freq_component, burst_hf_data, 10, 1)
        (score_non_burst, best_fit_non_burst, amp_signal_non_burst) = pac.run_dmi(low_freq_component, non_burst_hf_data, 10, 1)
        
        pickle.dump([best_fit_default, amp_signal_default, score_default,
                     best_fit_burst, amp_signal_burst, score_burst,
                     best_fit_non_burst, amp_signal_non_burst, score_non_burst], open(outpath + "data/test/" + file + ".pkl", "wb"))
        
    else:
        
        tmp = pickle.load(open(outpath + "data/2/" + file + ".pkl", "rb"))
        best_fit_default = tmp[0]; amp_signal_default = tmp[1]; score_default = tmp[2];
        best_fit_burst = tmp[3]; amp_signal_burst = tmp[4]; score_burst = tmp[5];
        best_fit_non_burst = tmp[6]; amp_signal_non_burst = tmp[7]; score_non_burst = tmp[8];
    
    if (visualize):
        (fig, axes) = plt.subplots(3, 1)
        axes[0].set_title("default: original data, score: %.2f" % (score_default,))
        axes[0].plot(best_fit_default)
        axes[0].plot(amp_signal_default)
        
        axes[1].set_title("burst: smoothed data, score: %.2f" % (score_burst,))
        axes[1].plot(best_fit_burst)
        axes[1].plot(amp_signal_burst)
        
        axes[2].set_title("non burst: smoothed data, score: %.2f" % (score_non_burst,))
        axes[2].plot(best_fit_non_burst)
        axes[2].plot(amp_signal_non_burst)
        
        fig.set_tight_layout(tight = True)
        fig.savefig(outpath + "img/2/" + file + image_format)
        
    return (score_default, score_burst, score_non_burst)

import finn.sfc.td as sfc_td

def get_dac(data, fs, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    
    if ((len(data)/fs) < 5):
        return None
    
    f_min = np.min([lf_f_min, hf_f_min])
    f_max = np.max([lf_f_max, hf_f_max])
       
    high_freq_component = ff.fir(np.asarray(data), 300, None, 0.1, fs)
    low_freq_component  = ff.fir(np.asarray(data), f_min, f_max, 0.1, fs)
    binarized_data = methods.detection.bursts.identify_peaks(np.copy(data), fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
    
    burst_data = transform_burst(np.copy(data), binarized_data, mode = "gaussian")
    non_burst_data = transform_non_burst(np.copy(data), binarized_data, mode = "gaussian")
    
    #--------------------- burst_data = np.abs(scipy.signal.hilbert(burst_data))
    #------------- non_burst_data = np.abs(scipy.signal.hilbert(non_burst_data))
    
    high_freq_component = ff.fir(np.asarray(high_freq_component), f_min, f_max, 0.1, fs)
    burst_data = ff.fir(np.asarray(burst_data), f_min, f_max, 0.1, fs)
    non_burst_data = ff.fir(np.asarray(non_burst_data), f_min, f_max, 0.1, fs)
    
    sfc_value_norm      = sfc_td.run_dac(low_freq_component, np.abs(scipy.signal.hilbert(high_freq_component)),
                                         f_min, f_max, fs, int(fs), int(fs), return_signed_conn = True, minimal_angle_thresh = 3)
    
    sfc_value_burst     = sfc_td.run_dac(low_freq_component, burst_data,
                                         f_min, f_max, fs, int(fs), int(fs), return_signed_conn = True, minimal_angle_thresh = 3)
    
    sfc_value_nburst    = sfc_td.run_dac(low_freq_component, non_burst_data,
                                         f_min, f_max, fs, int(fs), int(fs), return_signed_conn = True, minimal_angle_thresh = 3)
    
    print(sfc_value_norm[1], sfc_value_burst[1], sfc_value_nburst[1])
    
    return (sfc_value_norm[1], sfc_value_burst[1], sfc_value_nburst[1])

def amp_spike(data, fs, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    power_data = np.copy(data)
    power_data = np.abs(scipy.signal.hilbert(power_data))
    #power_data = ff.fir(power_data, hf_f_min, hf_f_max, 0.1, fs, 10e-5, 10e-7, int(fs), "zero", "fast"); power_data = np.abs(power_data)
    #power_data = np.abs(scipy.signal.hilbert(power_data))
    
    binarized_data = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh)
    
    data_mod = ff.fir(np.copy(data), 300, None, 1, fs)
    data_mod[data_mod > 0] = 0
    (spike_indices, _) = scipy.signal.find_peaks(np.abs(data_mod), height = peak_thresh)
    spikes = np.zeros(data.shape)
    spikes[spike_indices] = 1
    
    burst_amps = list()
    non_burst_amps = list()
    burst_spk_cnts = list()
    non_burst_spk_cnts = list()
    
    state = 0 if (binarized_data[0] == -1) else 1
    mrk = 0
    for idx in range(len(binarized_data)):
        if (binarized_data[idx] == 1 and state == 0):
            state = 1
            non_burst_amps.append(np.average(power_data[mrk:(idx - 1)]))
            non_burst_spk_cnts.append(np.sum(spikes[mrk:(idx - 1)]) * (fs/((idx - 1) - mrk)))
            mrk = idx
        if (binarized_data[idx] == -1 and state == 1):
            state = 0
            burst_amps.append(np.average(power_data[mrk:(idx - 1)]))
            burst_spk_cnts.append(np.sum(spikes[mrk:(idx - 1)]) * (fs/((idx - 1) - mrk)))
            mrk = idx
    
    win_sz = 2
    power_list = list()
    burst_spike_list = list()
    non_burst_spike_list = list()
    for idx in np.arange(0, len(data), fs * win_sz):
        start = int(idx)
        end = int(start + fs * win_sz)
        
        loc_power_data = np.average(power_data[start:end]); power_list.append(loc_power_data)
        loc_burst_spike = np.sum(spikes[start:end][(np.argwhere(binarized_data[start:end] == 1).squeeze())]) * fs/len(spikes[start:end][(np.argwhere(binarized_data[start:end] == 1).squeeze())]); burst_spike_list.append(loc_burst_spike)
        loc_non_burst_spike = np.sum(spikes[start:end][(np.argwhere(binarized_data[start:end] == -1).squeeze())]) * fs/len(spikes[start:end][(np.argwhere(binarized_data[start:end] == -1).squeeze())]); non_burst_spike_list.append(loc_non_burst_spike)
    
        print(loc_power_data, loc_burst_spike, loc_non_burst_spike, start, end)
    
    return (np.average(power_data), burst_amps, burst_spk_cnts, non_burst_amps, non_burst_spk_cnts, power_list, burst_spike_list, non_burst_spike_list)

def plot(data, fs, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    
    binarized_data = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh, buffer_area = "auto")
    lpf_data = ff.fir(data, 9, 50, 0.1, fs, 10e-5, 10e-7, int(fs), "zero", "fast")
    hpf_data = ff.fir(data, 300, None, 0.1, fs, 10e-5, 10e-7, int(fs), "zero", "fast")
    
    start = int(1 * fs)
    end = int(2.5 * fs)
    
    burst_data = np.copy(hpf_data)
    burst_data[np.argwhere(binarized_data != 1).squeeze()] = np.nan
    non_burst_data = np.copy(hpf_data)
    non_burst_data[np.argwhere(binarized_data != -1).squeeze()] = np.nan
    
    (fig, axes) = plt.subplots(3, 1)
    axes[0].plot(lpf_data[start:end], color = "black")
    axes[1].plot(burst_data[start:end], color = "black", zorder = 1); axes[1].plot(hpf_data[start:end], color = "grey", zorder = 0)
    axes[2].plot(non_burst_data[start:end], color = "black", zorder = 1); axes[2].plot(hpf_data[start:end], color = "grey", zorder = 0)
    
    axes[0].set_ylim((-.35, .35))
    axes[1].set_ylim((-2.5, 2.5))
    axes[2].set_ylim((-2.5, 2.5))
    
    (burst_hf_data, non_burst_hf_data) = preprocess_data(np.copy(data), fs, peak_spread, peak_thresh, mode = "gaussian")
    hpf_data = np.abs(scipy.signal.hilbert(hpf_data))
    
    (fig, axes) = plt.subplots(3, 1)
    axes[0].plot(scipy.signal.welch(lpf_data, fs, window = "hanning", nperseg = int(fs), noverlap = int(fs/2), nfft = int(fs), detrend = False, return_onesided = True)[1][2:45])
    axes[1].plot(scipy.signal.welch(burst_hf_data, fs, window = "hanning", nperseg = int(fs), noverlap = int(fs/2), nfft = int(fs), detrend = False, return_onesided = True)[1][2:45], color = "black", zorder = 1)
    axes[1].plot(scipy.signal.welch(hpf_data, fs, window = "hanning", nperseg = int(fs), noverlap = int(fs/2), nfft = int(fs), detrend = False, return_onesided = True)[1][2:45], color = "gray", zorder = 0)
    axes[2].plot(scipy.signal.welch(non_burst_hf_data, fs, window = "hanning", nperseg = int(fs), noverlap = int(fs/2), nfft = int(fs), detrend = False, return_onesided = True)[1][2:45], color = "black", zorder = 1)
    axes[2].plot(scipy.signal.welch(hpf_data, fs, window = "hanning", nperseg = int(fs), noverlap = int(fs/2), nfft = int(fs), detrend = False, return_onesided = True)[1][2:45], color = "gray", zorder = 0)
    
    plt.show(block = True)

def calculate_pac_exp_inner(loc_burst_hf_data, high_freq_component, fs_data01, f_offset, f_step_sz):
    #low_freq_component = ff.fir(np.copy(loc_burst_hf_data), 0.02 + f_offset, 0.2 + f_offset, 0.02, fs_data01)
    low_freq_component = ff.fir(np.copy(loc_burst_hf_data), 0.02 + f_offset, f_step_sz + f_offset, 0.1, fs_data01)
    (score_default, _, _) = pac.run_dmi(low_freq_component, high_freq_component, 10, 1)
    return score_default

def calculate_pac_exp(data, fs_data01, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):

    if (overwrite or os.path.exists(outpath + "data/5/" + file + ".pkl") == False):
        
        loc_data = ff.fir(np.copy(data), 300, None, 1, fs_data01)
        (loc_burst_hf_data, _) = preprocess_data(loc_data, fs_data01, peak_spread, peak_thresh)
        high_freq_component = ff.fir(loc_burst_hf_data, hf_f_min - 2, hf_f_max + 2, 0.1, fs_data01)
        
        low_freq_component = ff.fir(np.copy(data), 0.05, 3, 0.01, fs_data01)
        score = pac.run_dmi(low_freq_component, high_freq_component, 10, 1)[0]
                
        pickle.dump(score, open(outpath + "data/5/" + file + ".pkl", "wb"))
    else:
        score = pickle.load(open(outpath + "data/5/" + file + ".pkl", "rb"))
        
    return score
        
def calculate_pac_exp2(data, fs_data01, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):

    if (overwrite or os.path.exists(outpath + "data/6/" + file + ".pkl") == False):        
        high_freq_component = ff.fir(np.copy(data), lf_f_min - 2, lf_f_max + 2, 0.1, fs_data01)
        
        low_freq_component = ff.fir(np.copy(data), 0.05, 3, 0.01, fs_data01)
        score = pac.run_dmi(low_freq_component, high_freq_component, 10, 1)[0]
                
        pickle.dump(score, open(outpath + "data/6/" + file + ".pkl", "wb"))
    else:
        score = pickle.load(open(outpath + "data/6/" + file + ".pkl", "rb"))
        
    return score
        
def calculate_spectograms_inner(data, window_width, fs,
                                f_min, f_max, f_window_width, f_window_step_sz,
                                start_idx, filter_step_width = 1, peak_spread = 1.5, peak_thresh = 1.1):
    print("Processing idx %i of %i" % (start_idx, len(data) - window_width))
    data = np.asarray(data)
    
    
       
    loc_hf_data = ff.fir(np.copy(data), 300, None, filter_step_width, fs)
    #loc_lf_data = ff.fir(np.copy(data), filt_low, filt_high, filter_step_width, fs)
    
    (loc_burst_hf_data, loc_non_burst_hf_data) = preprocess_data(loc_hf_data, fs, peak_spread, peak_thresh)

    (bins, loc_lf_psd) = scipy.signal.welch(data, fs, window = "hann", nperseg = fs, noverlap = int(fs/2), nfft = fs, detrend = False, return_onesided = True)
    (_, loc_burst_hf_psd) = scipy.signal.welch(loc_burst_hf_data, fs, window = "hann", nperseg = fs, noverlap = int(fs/2), nfft = fs, detrend = False, return_onesided = True)
    (_, loc_non_burst_hf_psd) = scipy.signal.welch(loc_non_burst_hf_data, fs, window = "hann", nperseg = fs, noverlap = int(fs/2), nfft = fs, detrend = False, return_onesided = True)
    min_f_bin_idx = np.argmin(np.abs(bins - f_min))
    max_f_bin_idx = np.argmin(np.abs(bins - f_max))
    
    loc_burst_dmi_scores = list()
    loc_non_burst_dmi_scores = list()
    for f_idx in np.arange(f_min, f_max, f_window_step_sz):
        try:
            loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)        
            if ((loc_burst_hf_data != 0).all()):
                (loc_dmi_score, _, _) = pac.run_dmi(loc_dmi_lf_data, loc_burst_hf_data, 10, 1)
            else:
                loc_dmi_score = 0 #There cannot be any pac if there is no data in this segment
            loc_burst_dmi_scores.append(loc_dmi_score)
            
            loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)
            if ((loc_non_burst_hf_data != 0).all()):
                (loc_dmi_score, _, _) = pac.run_dmi(loc_dmi_lf_data, loc_non_burst_hf_data, 10, 1)
            else:
                loc_dmi_score = 0 #There cannot be any pac if there is no data in this segment
            loc_non_burst_dmi_scores.append(loc_dmi_score)
        except:
            print("A")
            
    return (loc_lf_psd[min_f_bin_idx:max_f_bin_idx], loc_burst_hf_psd[min_f_bin_idx:max_f_bin_idx], loc_non_burst_hf_psd[min_f_bin_idx:max_f_bin_idx],
            loc_burst_dmi_scores, loc_non_burst_dmi_scores)

import skimage.morphology

def calculate_spectograms(data, fs, peak_spread, peak_thresh, filter_step_width = 1,
                          lf_zoom_factor = 1, lf_offset_factor = 0, hf_zoom_factor = 1, hf_offset_factor = 0, pac_thresh = 0.2,
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (overwrite or os.path.exists(outpath + "data/3/" + file + ".pkl") == False):
        window_width = fs
        window_step_sz = int(fs/2)
        
        f_min = 12
        f_max = 45
        f_window_width = 2
        f_window_step_sz = 1
            
        tmp = np.asarray(tp.run(thread_cnt, calculate_spectograms_inner,
                                [(data[start_idx:(start_idx + window_width)], window_width, fs,
                                  f_min, f_max, f_window_width, f_window_step_sz,
                                  start_idx, filter_step_width, peak_spread, peak_thresh) for start_idx in np.arange(0, len(data) - window_width, window_step_sz)],
                                 max_time = None, delete_data = True))
            
        lf_psd = np.asarray(list(tmp[:, 0]))
        hf_burst_psd = np.asarray(list(tmp[:, 1]))
        hf_non_burst_psd = np.asarray(list(tmp[:, 2]))
        dmi_burst_scores = np.asarray(list(tmp[:, 3]))
        dmi_non_burst_scores = np.asarray(list(tmp[:, 4]))
        
        pickle.dump([lf_psd, hf_burst_psd, hf_non_burst_psd,
                     dmi_burst_scores, dmi_non_burst_scores], open(outpath + "data/3/" + file + ".pkl", "wb"))
        
    else:
        
        tmp = pickle.load(open(outpath + "data/3/" + file + ".pkl", "rb"))
        lf_psd = tmp[0]; hf_burst_psd = tmp[1]; hf_non_burst_psd = tmp[2];
        dmi_burst_scores = tmp[3]; dmi_non_burst_scores = tmp[4];
        
    
    lf_psd = np.flip(lf_psd, axis = 1)
    hf_burst_psd = np.flip(hf_burst_psd, axis = 1)
    hf_non_burst_psd = np.flip(hf_non_burst_psd, axis = 1)
    dmi_burst_scores = np.flip(dmi_burst_scores, axis = 1)
    dmi_non_burst_scores = np.flip(dmi_non_burst_scores, axis = 1)
    
    lf_psd = np.transpose(lf_psd)
    hf_burst_psd = np.transpose(hf_burst_psd)
    hf_non_burst_psd = np.transpose(hf_non_burst_psd)
    dmi_burst_scores = np.transpose(dmi_burst_scores)
    dmi_non_burst_scores = np.transpose(dmi_non_burst_scores)
    
    a = 1.7
    lf_psd = (np.power(a, np.log(lf_psd)) - 1)/(a-1)
    hf_burst_psd = (np.power(a, np.log(hf_burst_psd)) - 1)/(a-1)
    hf_non_burst_psd = (np.power(a, np.log(hf_non_burst_psd)) - 1)/(a-1)
    
    a = 1.7
    dmi_burst_scores = (np.power(a, dmi_burst_scores)-1)/(a - 1)
    dmi_non_burst_scores = (np.power(a, dmi_non_burst_scores)-1)/(a - 1)
            
    lf_psd = lf_psd - np.min(lf_psd); lf_psd = lf_psd / np.max(lf_psd)
    tmp = np.concatenate((hf_burst_psd, hf_non_burst_psd)).reshape(-1)
    hf_burst_psd = hf_burst_psd - np.min(tmp); hf_non_burst_psd = hf_non_burst_psd - np.min(tmp);
    tmp = np.concatenate((hf_burst_psd, hf_non_burst_psd)).reshape(-1)
    hf_burst_psd = hf_burst_psd / np.max(tmp); hf_non_burst_psd = hf_non_burst_psd - np.min(tmp); hf_non_burst_psd = hf_non_burst_psd / np.max(tmp)
    
    lf_psd = (lf_psd - 0.5)*2
    hf_burst_psd = (hf_burst_psd - 0.5)*2
    hf_non_burst_psd = (hf_non_burst_psd - 0.5)*2
    
    lf_psd = lf_psd * lf_zoom_factor
    lf_psd = lf_psd + lf_offset_factor
    hf_burst_psd = hf_burst_psd * hf_zoom_factor; hf_non_burst_psd = hf_non_burst_psd * hf_zoom_factor
    hf_burst_psd = hf_burst_psd + hf_offset_factor; hf_non_burst_psd = hf_non_burst_psd + hf_offset_factor
            
    (pac_burst_strength, pac_non_burst_strength, 
     pac_burst_specificity, pac_non_burst_specificity, 
     pac_burst_specific_strength, pac_non_burst_specific_strength, 
     mask_burst, mask_non_burst) = score_pac(lf_psd, hf_burst_psd, hf_non_burst_psd,
                                             dmi_burst_scores, dmi_non_burst_scores,
                                             2, 2, -2,
                                             3, pac_thresh)
    
    shuffled_mask_burst = np.copy(mask_burst); np.random.shuffle(shuffled_mask_burst)
    shuffled_mask_non_burst = np.copy(mask_non_burst); np.random.shuffle(shuffled_mask_non_burst)
    pac_random_burst_specific_strength = np.sum(np.multiply(dmi_burst_scores, shuffled_mask_burst))
    pac_random_non_burst_specific_strength = np.sum(np.multiply(dmi_non_burst_scores, shuffled_mask_non_burst))
    
    if (visualize):
        
        #=======================================================================
        # lf_psd = lf_psd[:, :10]
        # hf_burst_psd = hf_burst_psd[:, :10]
        # hf_non_burst_psd = hf_non_burst_psd[:, :10]
        # dmi_burst_scores = dmi_burst_scores[:, :10]
        # dmi_non_burst_scores = dmi_non_burst_scores[:, :10]
        #=======================================================================
        
        
        dmi_burst_scores[dmi_burst_scores < pac_thresh] = 0
        dmi_non_burst_scores[dmi_non_burst_scores < pac_thresh] = 0
        
        (fig, axes) = plt.subplots(3, 2)
        axes[0, 0].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -1, vmax = 1)
        axes[0, 1].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -1, vmax = 1)
        axes[1, 0].imshow(hf_burst_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -1, vmax = 1)
        axes[1, 1].imshow(hf_non_burst_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = -1, vmax = 1)
        axes[2, 0].imshow(dmi_burst_scores, vmin = 0.0, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        axes[2, 1].imshow(dmi_non_burst_scores, vmin = 0.0, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')

        axes[0, 0].set_title("burst: low frequency psd")
        axes[1, 0].set_title("burst: high frequency psd")
        axes[2, 0].set_title("burst: PAC: %2.2f" % (pac_burst_specificity, ))
        
        axes[0, 1].set_title("non burst: low frequency psd")
        axes[1, 1].set_title("non burst: high frequency psd")
        axes[2, 1].set_title("non burst: PAC: %2.2f" % (pac_non_burst_specificity, ))
        
        for ax in axes.reshape(-1):
            ax.set_yticks([0, 4, 9, 14, 19, 24, 29])
            ax.set_yticklabels(([15, 20, 25, 30, 35, 40, 45][::-1]))
        
        fig.set_tight_layout(tight = True)
        
        fig.savefig(outpath + "/img/3/" + file + image_format)
        plt.close()
        
    return (pac_burst_strength, pac_non_burst_strength,
            pac_burst_specificity, pac_non_burst_specificity, 
            pac_burst_specific_strength, pac_non_burst_specific_strength, 
            pac_random_burst_specific_strength, pac_random_non_burst_specific_strength)

#===============================================================================
# import finn.same_frequency_coupling.time_domain.directional_absolute_coherency as dac
# import finn.same_frequency_coupling.time_domain.magnitude_squared_coherency as msc
# 
# #===============================================================================
# # import scipy.signal
# # import mne.connectivity
# # import finn.same_frequency_coupling.__misc as misc
# # 
# # def calculate_sfc_inner(data, filter_step_width, fs, peak_spread, peak_thresh, nfft_fs_factor,
# #                         f_min_lf, f_max_lf,
# #                         f_min_hf, f_max_hf, f_step_size_hf,
# #                         start_idx, window_width):
# #     print("Processing idx %i of %i" % (start_idx, len(data) - window_width))
# #     data = np.asarray(data)
# #        
# #     loc_lf_data = ff.fir(np.copy(data), f_min_lf, f_max_lf, filter_step_width, fs)
# #     loc_hf_data = ff.fir(np.copy(data), 300, None, filter_step_width, fs)
# #     (loc_burst_hf_data, loc_non_burst_hf_data) = preprocess_data(loc_hf_data, fs, peak_spread, peak_thresh)
# #     loc_hf_data = np.abs(scipy.signal.hilbert(loc_hf_data))
# #     loc_burst_hf_data = np.abs(scipy.signal.hilbert(loc_burst_hf_data))
# #     loc_non_burst_hf_data = np.abs(scipy.signal.hilbert(loc_non_burst_hf_data))
# #     
# # #===============================================================================
# # #     # scipy.signal.coherence(loc_lf_data, loc_hf_data, fs, window = "hanning", nperseg = fs, noverlap = 0, nfft = fs, detrend = False)
# # #     #------------ msc.run(loc_lf_data, loc_hf_data, fs, nperseg = fs, nfft = fs)
# # # 
# # #     nperseg = int(fs/2)
# # # 
# # #     seg_data_X = misc.__segment_data(loc_hf_data, nperseg, "zero")
# # #     seg_data_Y = misc.__segment_data(loc_lf_data, nperseg, "zero")
# # # 
# # #     (bins, f_data_X) = misc.__calc_FFT(seg_data_X, fs, nfft = nperseg, window = "hanning")
# # #     (_,    f_data_Y) = misc.__calc_FFT(seg_data_Y, fs, nfft = nperseg, window = "hanning")
# # # 
# # #     s_xx = np.conjugate(f_data_X) * f_data_X * 2
# # #     s_yy = np.conjugate(f_data_Y) * f_data_Y * 2
# # #     s_xy = np.conjugate(f_data_X) * f_data_Y * 2
# # # 
# # #     s_xx = np.mean(s_xx, axis = 0)
# # #     s_yy = np.mean(s_yy, axis = 0)
# # #     s_xy = np.mean(s_xy, axis = 0)
# # # 
# # #     #Doublecheck s_xx, s_yy and s_xy after setting the scale to 1!
# # # 
# # #     mne_coh = mne.connectivity.spectral_connectivity(np.swapaxes(np.asarray([seg_data_X, seg_data_Y]), 0, 1), method = "coh", sfreq = fs, mode = "fourier")
# # #     finn_coh = msc.run(loc_lf_data, loc_hf_data, fs, nperseg, nperseg, False)[1]
# # #     scipy_coh = scipy.signal.coherence(loc_hf_data, loc_lf_data, fs, "hanning", nperseg, 0, nperseg, "constant")[1]
# # # 
# # #     (fig, axes) = plt.subplots(3, 1)
# # #     axes[0].plot(finn_coh[:50])
# # #     axes[1].plot(scipy_coh[:50])
# # #     axes[2].plot(np.square(np.concatenate(([0, 0, 0, 0, 0], mne_coh[0][1, 0][:45]))))
# # #     plt.show(block = False)
# # # 
# # #     print(bins[0:3])
# # #     (fig, axes) = plt.subplots(3, 1)
# # #     axes[0].plot(np.square(mne_coh[0][1, 0]))
# # #     axes[1].plot(finn_coh[5:])
# # #     axes[2].plot(scipy_coh[5:])
# # #     plt.show(block = True)
# # #     quit()
# # #===============================================================================
# # 
# #     dac_values = list()
# #     for f_start in np.arange(f_min_hf, f_max_hf, f_step_size_hf):
# #         full = msc.run(loc_lf_data, loc_hf_data, fs, int(fs*2), int(fs*2), True)
# #         full = dac.run(loc_lf_data, loc_hf_data, f_start, f_start + f_step_size_hf, fs, nperseg = int(fs/2),
# #                        nfft = int(fs/2), return_signed_conn = True, minimal_angle_thresh = 2, detrend = True)[1]
# #         burst = None
# #         non_burst = None
# #         
# #         dac_values.append([all, burst, non_burst])
# #             
# #     return dac_values
# # 
# # def calculate_sfc(data, fs, peak_spread, peak_thresh, filter_step_width = 1, nfft_fs_factor = 8,
# #                   f_min_lf = 0, f_max_lf = 0,
# #                   visualize = True, outpath = None, file = None, overwrite = True):
# #     
# #     if (overwrite or os.path.exists(outpath + "data/4/" + file + ".pkl") == False):
# #         window_width = int(fs * nfft_fs_factor)
# #         window_step_sz = int(fs/2)
# #         
# #         f_min_hf = 12
# #         f_max_hf = 45
# #         f_step_size_hf = 1
# #         
# #         start_idx = 0
# #         sfc_score = calculate_sfc_inner(data[start_idx:(start_idx + window_width)], filter_step_width, fs, peak_spread, peak_thresh, nfft_fs_factor, f_min_lf, f_max_lf, f_min_hf, f_max_hf, f_step_size_hf, start_idx, window_width)
# #             
# #         #-------- sfc_score = np.asarray(tp.run(thread_cnt, calculate_sfc_inner,
# #                                       # [(data[start_idx:(start_idx + window_width)], filter_step_width, fs, peak_spread, peak_thresh, nfft_fs_factor,
# #                                         # f_min_lf, f_max_lf, f_min_hf, f_max_hf, f_step_size_hf,
# #                                         # start_idx, window_width) for start_idx in np.arange(0, len(data) - window_width, window_step_sz)],
# #                                       #--- max_time = None, delete_data = True))
# #         
# #         pickle.dump(sfc_score, open(outpath + "data/4/" + file + ".pkl", "wb"))
# #     else:
# #         sfc_score = pickle.load(open(outpath + "data/4/" + file + ".pkl", "rb"))
# #     
# #     
# #         
# #     return sfc_score
# #===============================================================================
#===============================================================================

def score_pac(lf_psd, hf_burst_psd, hf_non_burst_psd,
              dmi_burst_scores, dmi_non_burst_scores, 
              inner_radius, outer_radius, uncertain_value,
              pac_morph_sz = 3, pac_thresh = 0.75):
    lf_psd = np.copy(lf_psd); hf_burst_psd = np.copy(hf_burst_psd); hf_non_burst_psd = np.copy(hf_non_burst_psd)
    dmi_burst_scores = np.copy(dmi_burst_scores); dmi_non_burst_scores = np.copy(dmi_non_burst_scores)

    (lf_psd, hf_burst_psd, hf_non_burst_psd, hf_burst_psd_fused, hf_non_burst_psd_fused) = get_full_reference_area(lf_psd, hf_burst_psd, hf_non_burst_psd, 
                                                                       inner_radius, outer_radius, uncertain_value)

    
    dmi_burst_scores_enlarged = skimage.morphology.closing(dmi_burst_scores, skimage.morphology.disk(pac_morph_sz))
    dmi_non_burst_scores_enlarged = skimage.morphology.closing(dmi_non_burst_scores, skimage.morphology.disk(pac_morph_sz))
    dmi_burst_scores_enlarged[dmi_burst_scores_enlarged < pac_thresh] = 0
    dmi_non_burst_scores_enlarged[dmi_non_burst_scores_enlarged < pac_thresh] = 0

    dmi_burst_scores_sng = np.sign(np.multiply(hf_burst_psd_fused, dmi_burst_scores_enlarged))
    dmi_non_burst_scores_sgn = np.sign(np.multiply(hf_non_burst_psd_fused, dmi_non_burst_scores_enlarged))
    
    #===========================================================================
    # (fig, axes) = plt.subplots(5, 2)
    # axes[0, 0].imshow(lf_psd, cmap='seismic', aspect = 'auto')
    # axes[0, 1].imshow(lf_psd, cmap='seismic', aspect = 'auto')
    # axes[1, 0].imshow(hf_burst_psd, vmin = -2, vmax = 2, cmap='seismic', aspect = 'auto')
    # axes[1, 1].imshow(hf_non_burst_psd, vmin = -2, vmax = 2, cmap='seismic', aspect = 'auto')
    # axes[2, 0].imshow(hf_burst_psd_fused, vmin = -2, vmax = 2, cmap='seismic', aspect = 'auto')
    # axes[2, 1].imshow(hf_non_burst_psd_fused, vmin = -2, vmax = 2, cmap='seismic', aspect = 'auto')
    # axes[3, 0].imshow(dmi_burst_scores, vmin = 0, vmax = 1, cmap='seismic', aspect = 'auto')
    # axes[3, 1].imshow(dmi_non_burst_scores, vmin = 0, vmax = 1, cmap='seismic', aspect = 'auto')
    # axes[4, 0].imshow(dmi_burst_scores_sng, cmap='seismic', aspect = 'auto')
    # axes[4, 1].imshow(dmi_non_burst_scores_sgn, cmap='seismic', aspect = 'auto')
    # fig.set_tight_layout(tight = True)
    # plt.show(block = True)
    #===========================================================================
    
    pac_burst_specificity = np.sum(dmi_burst_scores_sng[dmi_burst_scores_sng > 0])/np.sum(np.abs(dmi_burst_scores_sng)) if ((dmi_burst_scores_sng != 0).any()) else 0
    pac_non_burst_specificity = np.sum(dmi_non_burst_scores_sgn[dmi_non_burst_scores_sgn > 0])/np.sum(np.abs(dmi_non_burst_scores_sgn)) if ((dmi_non_burst_scores_sgn != 0).any()) else 0
    
    pac_burst_strength = np.sum(dmi_burst_scores)/dmi_burst_scores.shape[1]
    pac_non_burst_strength = np.sum(dmi_non_burst_scores)/dmi_non_burst_scores.shape[1]
    
    mask_burst = np.zeros(dmi_burst_scores_sng.shape); mask_burst[dmi_burst_scores_sng > 0] = 1; pac_burst_specific_strength = np.sum(np.multiply(mask_burst, dmi_burst_scores))
    mask_non_burst = np.zeros(dmi_non_burst_scores_sgn.shape); mask_non_burst[dmi_non_burst_scores_sgn > 0] = 1; pac_non_burst_specific_strength = np.sum(np.multiply(mask_non_burst, dmi_non_burst_scores))
    
    return (pac_burst_strength, pac_non_burst_strength, 
            pac_burst_specificity, pac_non_burst_specificity, 
            pac_burst_specific_strength, pac_non_burst_specific_strength, 
            mask_burst, mask_non_burst)

def get_full_reference_area(lf_psd, hf_burst_psd, hf_non_burst_psd, radius_small = 2, radius_large = 2, 
                            uncertain_value = 0):
    lf_psd = np.copy(lf_psd); hf_burst_psd = np.copy(hf_burst_psd); hf_non_burst_psd = np.copy(hf_non_burst_psd)    
    
    (lf_psd_small, hf_burst_psd_small, hf_non_burst_psd_small) = get_reference_area(lf_psd, hf_burst_psd, hf_non_burst_psd, radius_small)
    lf_psd_small = np.sign(lf_psd_small); lf_psd_small[lf_psd_small < 0] = 0
    hf_burst_psd_small = np.sign(hf_burst_psd_small); hf_burst_psd_small[hf_burst_psd_small < 0] = 0

    (lf_psd_large, hf_burst_psd_large, hf_non_burst_psd_large) = get_reference_area(lf_psd, hf_burst_psd, hf_non_burst_psd, radius_large)
    lf_psd_large = np.sign(lf_psd_large); lf_psd_large[lf_psd_large > 0] = 0
    hf_burst_psd_large = np.sign(hf_burst_psd_large); hf_burst_psd_large[hf_burst_psd_large > 0] = 0

    #===========================================================================
    # (_, axes) = plt.subplots(3, 1)
    # axes[0].imshow(lf_psd, cmap='seismic')
    # axes[1].imshow(hf_burst_psd, cmap='seismic')
    # axes[2].imshow(hf_non_burst_psd, cmap='seismic')
    # plt.show(block = True)
    #===========================================================================

    lf_psd = lf_psd_small + lf_psd_large
    hf_burst_psd = hf_burst_psd_small + hf_burst_psd_large
    hf_non_burst_psd = hf_non_burst_psd_small + hf_non_burst_psd_large
    
    #===========================================================================
    # (_, axes) = plt.subplots(3, 1)
    # axes[0].imshow(lf_psd, cmap='seismic')
    # axes[1].imshow(hf_burst_psd, cmap='seismic')
    # axes[2].imshow(hf_non_burst_psd, cmap='seismic')
    # plt.show(block = True)
    #===========================================================================
    
    hf_burst_psd_fused = lf_psd + hf_burst_psd; hf_burst_psd_fused[np.abs(hf_burst_psd_fused) < 1.5] = uncertain_value
    hf_non_burst_psd_fused = lf_psd + hf_non_burst_psd; hf_non_burst_psd_fused[np.abs(hf_non_burst_psd_fused) < 1.5] = uncertain_value
    
    #===========================================================================
    # (_, axes) = plt.subplots(3, 1)
    # axes[0].imshow(lf_psd, cmap='seismic')
    # axes[1].imshow(hf_burst_psd, vmin = -2, vmax = 2, cmap='seismic')
    # axes[2].imshow(hf_non_burst_psd, vmin = -2, vmax = 2, cmap='seismic')
    # plt.show(block = True)
    #===========================================================================
    
    hf_burst_psd_fused = hf_burst_psd_fused / 2
    hf_non_burst_psd_fused = hf_non_burst_psd_fused / 2
    
    return (lf_psd, hf_burst_psd, hf_non_burst_psd, hf_burst_psd_fused, hf_non_burst_psd_fused)

def get_reference_area(lf_psd, hf_burst_psd, hf_non_burst_psd, radius):
    #lf_psd = np.copy(lf_psd); hf_burst_psd = np.copy(hf_burst_psd); hf_non_burst_psd = np.copy(hf_non_burst_psd)
    
    lf_psd = skimage.morphology.dilation(lf_psd, skimage.morphology.disk(radius))
    hf_burst_psd = skimage.morphology.dilation(hf_burst_psd, skimage.morphology.disk(radius))
    hf_non_burst_psd = skimage.morphology.dilation(hf_non_burst_psd, skimage.morphology.disk(radius))
        
    return (lf_psd, hf_burst_psd, hf_non_burst_psd)

def normalize_data(data, min_val, max_val, radius = 3):
    data = data - min_val; data = data / (max_val - min_val)
    data = skimage.morphology.dilation(data, skimage.morphology.disk(radius))
    
    return data

def count_bursts(data, fs, peak_spread, peak_thresh, outpath, file):
    #===========================================================================
    # data = ff.fir(data, 300, None, 1, fs)
    # data = np.copy(data)
    # data[data > 0] = 0
    # 
    # (peaks, _) = scipy.signal.find_peaks(np.abs(data), height = peak_thresh)
    # peak_data = np.zeros(data.shape)
    # peak_data[peaks] = 1
    # binarized_data = methods.detection.bursts.identify_peaks(ff.fir(np.copy(np.asarray(data)), 300, None, 1, fs), fs, 300, None, peak_spread, peak_thresh)
    #===========================================================================
    
    in_data = ff.fir(np.copy(data), 300, None, 1, fs)
    in_data[in_data > 0] = 0
    
    (peaks, _) = scipy.signal.find_peaks(np.abs(in_data), height = peak_thresh)
    peak_data = np.zeros(in_data.shape)
    peak_data[peaks] = 1
    binarized_data = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh)
        
    
    burst_data = peak_data[np.argwhere(binarized_data == 1).squeeze()]
    non_burst_data = peak_data[np.argwhere(binarized_data == -1).squeeze()]
    
    burst_spikes_percentage = np.sum(burst_data) / np.sum(peak_data)
    non_burst_spikes_percentage = np.sum(non_burst_data) / np.sum(peak_data)
    
    spikes_per_second = np.sum(peak_data) / len(peak_data) * fs
    burst_spikes_per_second = np.sum(burst_data)/len(burst_data) * fs
    non_burst_spikes_per_second = np.sum(non_burst_data)/len(non_burst_data) * fs
    
    pickle.dump((burst_spikes_percentage, non_burst_spikes_percentage, burst_spikes_per_second, non_burst_spikes_per_second, spikes_per_second), open(outpath + "data/7/" + file + ".pkl", "wb"))
    print((burst_spikes_percentage, non_burst_spikes_percentage, burst_spikes_per_second, non_burst_spikes_per_second, spikes_per_second))
    
    return (float(burst_spikes_percentage), float(non_burst_spikes_percentage), float(burst_spikes_per_second), float(non_burst_spikes_per_second), float(spikes_per_second))

def ratio_burst_time(data, fs, peak_spread, peak_thresh):
    binarized_data = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh)
    
    return len(np.argwhere(binarized_data == 1).reshape(-1))/len(np.argwhere(binarized_data == -1).reshape(-1))

def get_hf_power(data, fs, peak_spread = 1.5, peak_thresh = 1.1):
    loc_data = np.copy(data)
    
    hpf_data01 = ff.fir(np.asarray(loc_data), 300, None, 1, fs)
    binarized_data = methods.detection.bursts.identify_peaks(hpf_data01, fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
    (peaks, _) = scipy.signal.find_peaks(np.abs(hpf_data01), height = peak_thresh)
    
    out_data_zero = np.zeros(loc_data.shape)
    out_data_zero[np.argwhere(binarized_data == 1).squeeze()] = loc_data[np.argwhere(binarized_data == 1).squeeze()]
    out_data_zero = np.abs(scipy.signal.hilbert(out_data_zero))
    
    out_data_trunc = loc_data[np.argwhere(binarized_data == 1).squeeze()]
    out_data_trunc = np.abs(scipy.signal.hilbert(out_data_trunc))
    
    nfft = int(fs / 10)
    (bins, out_data_zero_fd) = scipy.signal.welch(out_data_zero, fs = fs, window = "hanning",
                                                  nperseg = nfft, noverlap = nfft // 2, nfft = nfft,
                                                  detrend = "linear", return_onesided = True, scaling = "density", axis = -1, average = "mean")
    out_data_trunc_fd = scipy.signal.welch(out_data_trunc, fs = fs, window = "hanning",
                                           nperseg = nfft, noverlap = nfft // 2, nfft = nfft,
                                           detrend = "linear", return_onesided = True, scaling = "density", axis = -1, average = "mean")[1]
    
    plt.figure()
    plt.plot(bins[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))],
             out_data_zero_fd[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))], color = "blue")
    plt.plot(bins[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))],
             out_data_trunc_fd[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))], color = "green")
    
    loc_data = np.zeros(loc_data.shape)
    loc_data[peaks] += 1
    
    out_data_zero = np.zeros(loc_data.shape)
    out_data_zero[np.argwhere(binarized_data == 1).squeeze()] = loc_data[np.argwhere(binarized_data == 1).squeeze()]
    out_data_zero = np.abs(scipy.signal.hilbert(out_data_zero))
    
    out_data_trunc = loc_data[np.argwhere(binarized_data == 1).squeeze()]
    out_data_trunc = np.abs(scipy.signal.hilbert(out_data_trunc))
    
    nfft = int(fs / 10)
    (bins, out_data_zero_fd) = scipy.signal.welch(out_data_zero, fs = fs, window = "hanning",
                                                  nperseg = nfft, noverlap = nfft // 2, nfft = nfft,
                                                  detrend = "linear", return_onesided = True, scaling = "density", axis = -1, average = "mean")
    out_data_trunc_fd = scipy.signal.welch(out_data_trunc, fs = fs, window = "hanning",
                                           nperseg = nfft, noverlap = nfft // 2, nfft = nfft,
                                           detrend = "linear", return_onesided = True, scaling = "density", axis = -1, average = "mean")[1]
    
    plt.figure()
    plt.plot(bins[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))],
             out_data_zero_fd[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))], color = "blue")
    plt.plot(bins[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))],
             out_data_trunc_fd[np.argmin(np.abs(bins - 50)):np.argmin(np.abs(bins - 300))], color = "green")
    
    plt.show(block = True)

def __sine(x, phase, amp):
    """
    Internal method. Used in run_dmi to estimate the direct modulation index. The amount of PAC is quantified via a sine fit. This sine is defined by the following paramters:
    
    :param x: Samples
    :param phase: Phase shift of the sine.
    :param amp: Amplitude of the sine.
    
    :return: Returns the fitted sine at the locations indicated by x.
    """
    freq = 1
    fs = 1
    return amp * (np.sin(2 * np.pi * freq * (x - phase) / fs))

import scipy.stats

def get_phase_spike(data, fs, peak_spread = 1.5, peak_thresh = 1.1, phase_window_half_sz = 5, max_model_fit_iterations = 300):
    loc_data = np.copy(data)
    
    spike_data01 = ff.fir(np.asarray(loc_data), 300, None, 1, fs)
    (peaks, _) = scipy.signal.find_peaks(np.abs(spike_data01), height = peak_thresh)
    spikes = np.zeros(loc_data.shape)
    spikes[peaks] = 1
    
    beta_data = ff.fir(np.copy(loc_data), 12, 32, 1, fs, 10e-5, 10e-7, fs, "zero", "fast")
    beta_amp = np.abs(beta_data)
    
    (score, best_fit, amplitude_signal) = tf.dmi(np.copy(beta_data), np.copy(spikes), phase_window_half_size = 5, max_model_fit_iterations = 300)
    amplitude_signal -= min(amplitude_signal)
    
    mrk = np.zeros(beta_data.shape)
    phase_signal = np.angle(scipy.signal.hilbert(beta_data), deg = True)
    mrk[phase_signal > -135] = 1
    mrk[phase_signal >= -45] = 0
    mrk = np.concatenate((mrk, [0]))
    
    plt.figure()
    plt.plot(mrk * np.max(phase_signal))
    plt.plot(phase_signal)
    
    corr_spikes = list()
    corr_amp = list()
    area_sz = 0
    for idx in range(len(mrk)):
        if (mrk[idx] == 1):
            area_sz += 1
        if (mrk[idx] == 0):
            if (area_sz > 10):
                corr_spikes.append(np.sum(spikes[(idx - area_sz):(idx - 1)]))
                corr_amp.append(np.sum(beta_amp[(idx - area_sz):(idx - 1)]))

            area_sz = 0
    corr_spikes = np.asarray(corr_spikes)
    corr_amp = np.asarray(corr_amp)
    
    corr_spikes_0 = corr_spikes[corr_spikes == 0]; corr_spikes_1 = corr_spikes[corr_spikes == 1]; corr_spikes_2 = corr_spikes[corr_spikes == 2]
    corr_amp_0 = corr_amp[corr_spikes == 0]; corr_amp_1 = corr_amp[corr_spikes == 1]; corr_amp_2 = corr_amp[corr_spikes == 2]
    
    print(scipy.stats.ttest_ind(corr_amp_0, corr_amp_1))
    print(scipy.stats.ttest_ind(corr_amp_1, corr_amp_2))
    print(scipy.stats.ttest_ind(corr_amp_0, corr_amp_2))
    
    plt.figure()
    plt.boxplot([corr_amp_0, corr_amp_1, corr_amp_2], labels = ["0 spikes", "1 spike", "2 spikes"])
#    plt.scatter(corr_spikes, corr_amp)
    plt.show(block = True)
            
    plt.figure()
    smooth_sz = 5
    smooth_beta_hist = pandas.Series(np.pad(amplitude_signal, smooth_sz, "constant", constant_values = [0])).rolling(window = smooth_sz, center = True).mean()[smooth_sz:-smooth_sz]
    plt.bar(np.arange(-180 + phase_window_half_sz, 180 - phase_window_half_sz, phase_window_half_sz * 2), smooth_beta_hist, width = phase_window_half_sz * 2 - 1)
    best_fit *= (np.max(smooth_beta_hist) - np.min(smooth_beta_hist)) / 2
    best_fit += (np.max(smooth_beta_hist) + np.min(smooth_beta_hist)) / 2
    plt.plot(np.arange(-180 + phase_window_half_sz, 180 - phase_window_half_sz, phase_window_half_sz * 2), best_fit, color = "red")
    plt.title(str(score))
    
    plt.show(block = True)

def get_phase_data(data, fs, peak_spread = 1.5, peak_thresh = 1.1, phase_window_half_sz = 5, max_model_fit_iterations = 300, file = None,
           visualize_phase_effects = True, visualize_binned_effects = False):
    loc_data = np.copy(data)
    
    spike_data01 = ff.fir(np.asarray(loc_data), 300, None, 1, fs)
    hfo_data = ff.fir(np.asarray(loc_data), 150, 300, 1, fs)
    hfo_data = np.abs(scipy.signal.hilbert(hfo_data))
    (peaks, _) = scipy.signal.find_peaks(np.abs(spike_data01), height = peak_thresh)
    spikes = np.zeros(loc_data.shape)
    spikes[peaks] = 1
    
    beta_data = ff.fir(np.copy(loc_data), 12, 32, 1, fs, 10e-5, 10e-7, fs, "zero", "fast")
    beta_amp = np.abs(beta_data)
    
    (_, _, spike_amplitude_signal) = tf.dmi(np.copy(beta_data), np.copy(spikes), phase_window_half_size = 5, max_model_fit_iterations = 300)
    (score, best_fit, hfo_amplitude_signal) = tf.dmi(np.copy(beta_data), np.copy(hfo_data), phase_window_half_size = 5, max_model_fit_iterations = 300)
    spike_amplitude_signal -= min(spike_amplitude_signal)
    hfo_amplitude_signal -= min(hfo_amplitude_signal)
    
    print(scipy.stats.spearmanr(best_fit, spike_amplitude_signal))
    print(scipy.stats.spearmanr(best_fit, hfo_amplitude_signal))
    
    if (visualize_phase_effects):            
        plt.figure()
        smooth_sz = 5
        spike_smooth_beta_hist = pandas.Series(np.pad(spike_amplitude_signal, smooth_sz, "constant", constant_values = [0])).rolling(window = smooth_sz, center = True).mean()[smooth_sz:-smooth_sz]
        hfo_smooth_beta_hist = pandas.Series(np.pad(hfo_amplitude_signal, smooth_sz, "constant", constant_values = [0])).rolling(window = smooth_sz, center = True).mean()[smooth_sz:-smooth_sz]
        spike_smooth_beta_hist /= np.max(spike_smooth_beta_hist)
        hfo_smooth_beta_hist /= np.max(hfo_smooth_beta_hist)
        best_fit /= np.max(best_fit)
        
        plt.bar(np.arange(-180 + phase_window_half_sz, 180 - phase_window_half_sz, phase_window_half_sz * 2), hfo_smooth_beta_hist, width = phase_window_half_sz * 2 - 1)
        plt.bar(np.arange(-180 + phase_window_half_sz, 180 - phase_window_half_sz, phase_window_half_sz * 2), spike_smooth_beta_hist, width = (phase_window_half_sz * 2 - 1) // 2)
        best_fit *= (np.max(spike_smooth_beta_hist) - np.min(spike_smooth_beta_hist)) / 2
        best_fit += (np.max(spike_smooth_beta_hist) + np.min(spike_smooth_beta_hist)) / 2
        plt.plot(np.arange(-180 + phase_window_half_sz, 180 - phase_window_half_sz, phase_window_half_sz * 2), best_fit, color = "red")
        plt.title(str(score))
    
        plt.show(block = True)
    
    mrk = np.zeros(beta_data.shape)
    phase_signal = np.angle(scipy.signal.hilbert(beta_data), deg = True)
    mrk[phase_signal > -135] = 1
    mrk[phase_signal >= -45] = 0
    mrk = np.concatenate((mrk, [0]))
    binarized_data = methods.detection.bursts.identify_peaks(data, fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
    
    corr_spikes = list()
    corr_beta_amp = list()
    corr_hfo_amp = list()
    burst_type = list()
    spike_area_sz = 0
    for idx in range(len(mrk)):
        if (mrk[idx] == 1):
            spike_area_sz += 1
        if (mrk[idx] == 0):
            if (spike_area_sz > 10):
                corr_spikes.append(np.sum(spikes[(idx - spike_area_sz):(idx - 1)]))
                corr_beta_amp.append(np.sum(beta_amp[(idx - spike_area_sz):(idx - 1)]))
                corr_hfo_amp.append(np.sum(hfo_data[(idx - spike_area_sz):(idx - 1)]))
                burst_type.append(1 if (np.average(binarized_data[(idx - spike_area_sz):(idx - 1)]) > 0.5) else 0)

            spike_area_sz = 0
    corr_spikes = np.asarray(corr_spikes)
    corr_beta_amp = np.asarray(corr_beta_amp)
    corr_hfo_amp = np.asarray(corr_hfo_amp)
    burst_type = np.asarray(burst_type)
    
    corr_beta_amp_0 = corr_beta_amp[corr_spikes == 0]; corr_beta_amp_1 = corr_beta_amp[corr_spikes == 1]; corr_beta_amp_2 = corr_beta_amp[corr_spikes == 2]
    corr_hfo_amp_0 = corr_hfo_amp[corr_spikes == 0]; corr_hfo_amp_1 = corr_hfo_amp[corr_spikes == 1]; corr_hfo_amp_2 = corr_hfo_amp[corr_spikes == 2]
    burst_type_0 = burst_type[corr_spikes == 0]; burst_type_1 = burst_type[corr_spikes == 1]; burst_type_2 = burst_type[corr_spikes == 2]

    if (visualize_binned_effects):
        plt.figure()
        plt.boxplot([corr_beta_amp_0, corr_beta_amp_1, corr_beta_amp_2], labels = ["0 spikes", "1 spike", "2 spikes"])
        plt.figure()
        plt.boxplot([corr_hfo_amp_0, corr_hfo_amp_1, corr_hfo_amp_2], labels = ["0 spikes", "1 spike", "2 spikes"])
    #    plt.scatter(corr_spikes, corr_amp)
        plt.show(block = True)
    
    os.makedirs("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/", exist_ok = True)
    
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_beta_amp_0.npy", corr_beta_amp_0)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_beta_amp_1.npy", corr_beta_amp_1)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_beta_amp_2.npy", corr_beta_amp_2)
    
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_hfo_amp_0.npy", corr_hfo_amp_0)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_hfo_amp_1.npy", corr_hfo_amp_1)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/corr_hfo_amp_2.npy", corr_hfo_amp_2)
    
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/burst_type_0.npy", burst_type_0)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/burst_type_1.npy", burst_type_1)
    np.save("/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/" + file + "/burst_type_2.npy", burst_type_2)

import analysis.beta.to_finn as tf
import lmfit

def absolute_pac(data, fs, peak_spread, peak_thresh,
                  lf_f_min, lf_f_max, hf_f_min, hf_f_max,
                  visualize = True, outpath = None, file = None, overwrite = True):
    """
    Determines the pac between the beta signal and the burst signal when the latter is transformed to (0, 1)
    """
    
    high_freq_component = ff.fir(np.asarray(data), 300, None, 0.1, fs)
    low_freq_component = ff.fir(np.asarray(data), hf_f_min, hf_f_max, 0.1, fs)
        
    binarized_data = methods.detection.bursts.identify_peaks(high_freq_component, fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
    binarized_data[binarized_data == 0] = np.nan
    binarized_data[binarized_data == -1] = 0
    
    
    (score_burst, best_fit_burst, amp_signal_burst) = pac.run_dmi(low_freq_component, binarized_data, 10, 1)
    
    if (visualize):
        plt.figure()
        plt.plot(high_freq_component, color = "black")
        plt.plot(binarized_data * np.max(high_freq_component), color = "blue")
        plt.figure()
        plt.plot(np.arange(-180, 181, 1), amp_signal_burst)
        plt.plot(np.arange(-180, 181, 1), best_fit_burst)
        
        plt.show(block = True)
    
    pickle.dump([score_burst, best_fit_burst, amp_signal_burst], open(outpath + "data/absolute_pac/" + file + ".pkl", "wb"))

def main(mode = "power", overwrite = False, visualize = False):
    meta_file = methods.data_io.ods.ods_data("../../../../data/meta.ods")
    meta_data = meta_file.get_sheet_as_dict("beta")
    in_path = "../../../../data/beta/data_for_python/"
    out_path = "../../../../results/beta/"
    
    dac_data = list()
    amp_spike_data = list()
    
    for (file_idx, file) in enumerate(meta_data["file"]):
        
        if (file == ""):
            continue
        
        if (file != "2622_s1_922-BETA" and file != "2623-s3-666"):
            continue
        
        #=======================================================================
        # if (file != "2622_s1_614b-BETA" and file != "2622_s1_763-BETA" and file != "2622_s1_1017b-BETA-two-beta-neurons-synchronous-longer-file"
        #     and file != "2622_s2_598-BETA"):
        #     continue
        #=======================================================================
        
        #=======================================================================
        # if (file != "2622_s1_614b-BETA" and file != "2622_s1_763-BETA" and file != "2622_s1_1017b-BETA-two-beta-neurons-synchronous-longer-file"
        #     and file != "2622_s2_598-BETA" and file != "2779_s1_685-BETA-faster" and file != "2884_s1_596_BETA" and file != "2903-s2-650"
        #     and file != "2903-s2-775-long" and file != "3304_tbd_s1_138" and file != "3304_tbd_s1_497" and file != "3304_tbd_s2_440"
        #     and file != "2559_s1_602-BETA"):
        #     continue
        #=======================================================================
        
        #=======================================================================
        # if (file_idx != 49):
        #     continue
        #=======================================================================
        
        if (int(meta_data["valid_data"][file_idx]) == 0 or int(meta_data["process data"][file_idx]) == 0):
            continue
        
        print("file", file)
        
        if (overwrite == True):
            file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
            file_data = pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))
            
            fs_data01 = int(file_hdr[20]['fs'])
            
            
            loc_data = np.asarray(file_data[20])
            if ("exp pac" not in mode):
                loc_data = ff.fir(loc_data, 2, None, 0.1, fs_data01)
            #loc_data = ff.fir(np.asarray(file_data[20]), 12, None, 0.1, fs_data01)
        else:
            file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
            file_data = None
            fs_data01 = int(file_hdr[20]['fs'])
            loc_data = None
                    
        lf_f_min = int(meta_data["lf f min"][file_idx])
        lf_f_max = int(meta_data["lf f max"][file_idx])
        hf_f_min = int(meta_data["hf f min"][file_idx])
        hf_f_max = int(meta_data["hf f max"][file_idx])            
      
        if ("power" in mode):
            beta_values = plot_hf_lf_components(loc_data, fs_data01, lf_f_min, lf_f_max, hf_f_min, hf_f_max, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]), outpath = out_path, file = file,
                                                overwrite = overwrite, visualize = visualize)
            meta_data["beta lfp strength 1"][file_idx] = float(beta_values[0])
            meta_data["beta overall strength 1"][file_idx] = float(beta_values[1])
            meta_data["beta burst strength 1"][file_idx] = float(beta_values[2])
            meta_data["beta non burst strength 1"][file_idx] = float(beta_values[3])
                        
        if ("overall pac" in mode):
            pac_values = calculate_pac(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                                       lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                                       hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                                       outpath = out_path, file = file,
                                       overwrite = overwrite, visualize = visualize)
            meta_data["pac overall strength 2"][file_idx] = float(pac_values[0])
            meta_data["pac burst strength 2"][file_idx] = float(pac_values[1])
            meta_data["pac non burst strength 2"][file_idx] = float(pac_values[2])
            
        if("specific pac" in mode):
            pac_values = calculate_spectograms(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]), 
                                               lf_zoom_factor = float(meta_data["lf_zoom_factor"][file_idx]), lf_offset_factor = float(meta_data["lf_offset_factor"][file_idx]),
                                               hf_zoom_factor = float(meta_data["hf_zoom_factor"][file_idx]), hf_offset_factor = float(meta_data["hf_offset_factor"][file_idx]), pac_thresh = float(meta_data["pac_thresh"][file_idx]),
                                               outpath = out_path, file = file, overwrite = overwrite, visualize = visualize)
            meta_data["pac burst strength 3"][file_idx] = float(pac_values[0])
            meta_data["pac non burst strength 3"][file_idx] = float(pac_values[1])
            meta_data["pac burst specificity 3"][file_idx] = float(pac_values[2])
            meta_data["pac non burst specificity 3"][file_idx] = float(pac_values[3])
            meta_data["pac burst specific strength 3"][file_idx] = float(pac_values[4])
            meta_data["pac non burst specific strength 3"][file_idx] = float(pac_values[5])
            meta_data["pac random burst specific strength 3"][file_idx] = float(pac_values[6])
            meta_data["pac random non burst specific strength 3"][file_idx] = float(pac_values[7])
            ref_value = np.max([pac_values[4], pac_values[5], pac_values[6], pac_values[7]])
            meta_data["pac burst specific strength norm 3"][file_idx] = float(pac_values[4]/ref_value) if (ref_value != 0) else 0
            meta_data["pac non burst specific strength norm 3"][file_idx] = float(pac_values[5]/ref_value) if (ref_value != 0) else 0
            meta_data["pac random burst specific strength norm 3"][file_idx] = float(pac_values[6]/ref_value) if (ref_value != 0) else 0
            meta_data["pac random non burst specific strength norm 3"][file_idx] = float(pac_values[7]/ref_value) if (ref_value != 0) else 0
        
        if ("cnt_burst" in mode):
            score = count_bursts(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]),
                                 peak_thresh = float(meta_data["peak_thresh"][file_idx]), outpath = out_path, file = file)
            
            meta_data["spikes_within"][file_idx] = score[0]
            meta_data["spikes_outside"][file_idx] = score[1] 
            meta_data["spikes_within_per_second"][file_idx] = score[2]
            meta_data["spikes_outside_per_second"][file_idx] = score[3]
            meta_data["spikes_per_second"][file_idx] = score[4]
        
        if ("exp pac" in mode):
            score = calculate_pac_exp(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                                      lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                                      hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                                      outpath = out_path, file = file,
                                      overwrite = overwrite, visualize = visualize)
            meta_data["reverse hf pac score"][file_idx] = float(score)
        
        if ("exp pac 2" in mode):
            score = calculate_pac_exp2(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                                       lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                                       hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                                       outpath = out_path, file = file,
                                       overwrite = overwrite, visualize = visualize)
            meta_data["reverse lf pac score"][file_idx] = float(score)
            
        
        if ("ratio burst time" in mode):
            score = ratio_burst_time(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]),
                                 peak_thresh = float(meta_data["peak_thresh"][file_idx]))
            meta_data["ratio_of_burst_time"][file_idx] = float(score)
            
        
        if ("hf power" in mode):
            get_hf_power(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]),
                         peak_thresh = float(meta_data["peak_thresh"][file_idx]))
            
        if ("phase spike" in mode):
            get_phase_spike(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), 
                            peak_thresh = float(meta_data["peak_thresh"][file_idx]))
        
        if ("phase data" in mode):
            get_phase_data(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]),
                           peak_thresh = float(meta_data["peak_thresh"][file_idx]), file = file)
            
        if ("absolute_pac" in mode):
            absolute_pac(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                         lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                         hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                         outpath = out_path, file = file,
                         overwrite = overwrite, visualize = visualize)
            
        if ("default_beta_phase_pac" in mode):
            default_beta_phase_pac(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                                   lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                                   hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                                   outpath = out_path, file = file,
                                   overwrite = overwrite, visualize = visualize)
            
        if ("dac" in mode):
            tmp = get_dac(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                       lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                       hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                       outpath = out_path, file = file,
                       overwrite = overwrite, visualize = visualize)
            if (tmp is not None):
                dac_data.append([tmp[0], tmp[1], tmp[2], meta_data["lf auto"][file_idx], meta_data["hf auto"][file_idx], file])
        
        if ("amp_spike") in mode:
            loc_amp_spike_data = amp_spike(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                                           lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                                           hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                                           outpath = out_path, file = file,
                                           overwrite = overwrite, visualize = visualize)
            amp_spike_data.append((*loc_amp_spike_data, file))
        
        if ("plot" in mode):
            plot(loc_data, fs_data01, peak_spread = float(meta_data["peak_spread"][file_idx]), peak_thresh = float(meta_data["peak_thresh"][file_idx]),
                 lf_f_min = float(meta_data["lf f min"][file_idx]), lf_f_max = float(meta_data["lf f max"][file_idx]),
                 hf_f_min = float(meta_data["hf f min"][file_idx]), hf_f_max = float(meta_data["hf f max"][file_idx]),
                 outpath = out_path, file = file,
                 overwrite = overwrite, visualize = visualize)
            
        if (len(mode) > 1):
            meta_file.modify_sheet_from_dict("beta", meta_data)
            meta_file.write_file()
        
        
        plt.close("all")
    if ("dac" in mode):
        np.save("dac.npy", np.asarray(dac_data))
    if ("amp_spike") in mode:
        np.save("amp_spike_data.npy", np.asarray(amp_spike_data))
    
    if (len(mode) != 3):
        meta_file.modify_sheet_from_dict("beta", meta_data)
        meta_file.write_file()
    print("Terminated successfully")
    
#===============================================================================
# #main(["power", "overall pac", "specific pac"], overwrite = False, visualize = True)
# #main(["overall pac"], overwrite = True, visualize = True)
# #main(["specific pac"], overwrite = True, visualize = True)
# #main(["specific pac"], overwrite = False, visualize = True)
# #main(["power"], overwrite = True, visualize = True)
# #main(["phase spike"], overwrite = True, visualize = True)
# 
# #main(["phase data"], overwrite = True, visualize = True)
# #main(["absolute_pac"], overwrite = True, visualize = False)
# #main(["dac"], overwrite = True, visualize = False)
#===============================================================================

#main(["power"], overwrite = True, visualize = True)
#main(["overall pac"], overwrite = True, visualize = True)
#main(["specific pac"], overwrite = False, visualize = True)
#main(["cnt_burst"], overwrite = True, visualize = True)
#main(["dac"], overwrite = True, visualize = True)
main(["test"], overwrite = True, visualize = True)


