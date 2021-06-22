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
import finn.cross_frequency_coupling.direct_modulation_index as dmi

import finn.misc.timed_pool as tp

import methods.detection.bursts
import methods.data_io.ods

import os
import scipy.stats

thread_cnt = 11
image_format = ".png"
#image_format = ".svg"

def preprocess_data(in_data, fs, peak_spread, peak_thresh):
    in_data = np.copy(in_data)
    
    burst_data = transform_burst(in_data, fs, 70, None, peak_spread, peak_thresh)
    non_burst_data = transform_non_burst(in_data, fs, 70, None, peak_spread, peak_thresh)
    
    return (burst_data, non_burst_data)

def transform_burst(in_data, fs, filter_f_min, filter_f_max, peak_spread, peak_thresh):
    binarized_data = methods.detection.bursts.identify_peaks(in_data, fs, filter_f_min, filter_f_max, peak_spread, peak_thresh)
            
    out_data = np.zeros(in_data.shape)
    out_data[np.argwhere(binarized_data == 1).squeeze()] = in_data[np.argwhere(binarized_data == 1).squeeze()]
    
    out_data = np.abs(scipy.signal.hilbert(out_data))
    
    return out_data

def transform_non_burst(in_data, fs, filter_f_min, filter_f_max, peak_spread, peak_thresh):
    binarized_data = methods.detection.bursts.identify_peaks(in_data, fs, filter_f_min, filter_f_max, peak_spread, peak_thresh)
        
    out_data = np.zeros(in_data.shape)
    out_data[np.argwhere(binarized_data == 0).squeeze()] = in_data[np.argwhere(binarized_data == 0).squeeze()]
    
    out_data = np.abs(scipy.signal.hilbert(out_data))
    
    return out_data

def plot_hf_lf_components(data, fs_data01, lf_min_f, lf_max_f, hf_min_f, hf_max_f, f_min = 2, f_max = 12, peak_spread = 1.5, peak_thresh = 1.1, 
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (fs_data01 is not None):
        if (len(data) < fs_data01):
            print("Insufficient data")
            return (-1, -1, -1, -1)
    
    if (overwrite or os.path.exists(outpath + "data/1/" + file + ".pkl") == False):
        # filtering
        lpf_data01 = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        bpf_data01 = ff.fir(np.asarray(data), lf_min_f, lf_max_f, 0.1, fs_data01)
        hpf_data01 = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
        
        # spike signal smooth-transform
        (burst_hf_data, non_burst_hf_data) = preprocess_data(hpf_data01, fs_data01, peak_spread, peak_thresh)        
            
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
     
    ref_f_width = 2
    ref_lf_f_min = lf_min_f-ref_f_width if (lf_min_f-ref_f_width > 1) else 2
    ref_hf_f_min = hf_min_f-ref_f_width if (hf_min_f-ref_f_width > 1) else 2
    
    #------------------------------------------------- print(lf_min_f, lf_max_f)
    #------------------------------------------------- print(hf_min_f, hf_max_f)
    
    lf_tremor_strength = np.average(lpf_psd[lf_min_f:lf_max_f]) / np.average([np.average(lpf_psd[(ref_lf_f_min):lf_min_f]), np.average(lpf_psd[lf_max_f:(lf_max_f+ref_f_width)])])
    #lf_tremor_strength = np.average(lpf_psd[lf_min_f:lf_max_f]) / np.average(lpf_psd[lf_max_f:(lf_max_f+ref_f_width)])
    hf_tremor_strength = np.average(raw_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(raw_hf_data_psd[(ref_hf_f_min):hf_min_f]), np.average(raw_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    #hf_tremor_strength = np.average(raw_hf_data_psd[hf_min_f:hf_max_f]) / np.average(raw_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])
    burst_tremor_strength = np.average(burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(burst_hf_data_psd[(ref_hf_f_min):hf_min_f]), np.average(burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    #burst_tremor_strength = np.average(burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average(raw_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])
    #burst_tremor_strength = np.average(burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average(burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])
    non_burst_tremor_strength = np.average(non_burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average([np.average(non_burst_hf_data_psd[(ref_hf_f_min):hf_min_f]), np.average(non_burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])])
    #non_burst_tremor_strength = np.average(non_burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average(raw_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])
    #non_burst_tremor_strength = np.average(non_burst_hf_data_psd[hf_min_f:hf_max_f]) / np.average(non_burst_hf_data_psd[hf_max_f:(hf_max_f+ref_f_width)])
    
    lf_tremor_strength = np.log(lf_tremor_strength)
    hf_tremor_strength = np.log(hf_tremor_strength)
    burst_tremor_strength = np.log(burst_tremor_strength)
    non_burst_tremor_strength = np.log(non_burst_tremor_strength)
    
    if (visualize):
        (fig, axes) = plt.subplots(3, 2)
        axes[0, 0].plot(t_data01, lpf_data01, color = "blue")
        axes[0, 0].plot(t_data01, bpf_data01, color = "orange")
        axes[1, 0].plot(t_data01, hpf_data01, color = "blue")
        axes[1, 0].plot(t_data01, burst_hf_data, color = "orange")
        axes[2, 0].plot(t_data01, hpf_data01, color = "blue")
        axes[2, 0].plot(t_data01, non_burst_hf_data, color = "orange")
        
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
        axes[0, 1].set_title("default %1.2f" % (lf_tremor_strength,))
        axes[1, 1].set_title("burst %1.2f of %1.2f" % (burst_tremor_strength, hf_tremor_strength))
        axes[2, 1].set_title("non burst %1.2f of %1.2f" % (non_burst_tremor_strength, hf_tremor_strength))
        
        axes[1, 1].set_yticks([0, vmax])
        axes[2, 1].set_yticks([0, vmax])
        
        axes[0, 1].set_xticks(np.arange(f_min, f_max, 2))
        axes[1, 1].set_xticks(np.arange(f_min, f_max, 2))
        axes[2, 1].set_xticks(np.arange(f_min, f_max, 2))
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticks(), rotation = 45)
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticks(), rotation = 45)
        axes[2, 1].set_xticklabels(axes[2, 1].get_xticks(), rotation = 45)
        
        fig.set_tight_layout(tight = True)
        fig.savefig(outpath + "img/1/" + file + image_format)
            
    return (lf_tremor_strength, hf_tremor_strength, burst_tremor_strength, non_burst_tremor_strength)

def calculate_dmi(data, fs_data01, peak_spread, peak_thresh,
                  visualize = True, outpath = None, file = None, overwrite = True):
    if (overwrite or os.path.exists(outpath + "data/2/" + file + ".pkl") == False):
    
        high_freq_component = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
        #low_freq_component = ff.fir(np.asarray(data), filt_low, filt_high, 0.1, fs_data01)
        low_freq_component = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        
        (burst_hf_data, non_burst_hf_data) = preprocess_data(high_freq_component, fs_data01, peak_spread, peak_thresh)
        
        (score_default, best_fit_default, amp_signal_default) = dmi.run(low_freq_component, high_freq_component, 10, 1)
        (score_burst, best_fit_burst, amp_signal_burst) = dmi.run(low_freq_component, burst_hf_data, 10, 1)
        (score_non_burst, best_fit_non_burst, amp_signal_non_burst) = dmi.run(low_freq_component, non_burst_hf_data, 10, 1)
        
        pickle.dump([best_fit_default, amp_signal_default, score_default,
                     best_fit_burst, amp_signal_burst, score_burst,
                     best_fit_non_burst, amp_signal_non_burst, score_non_burst], open(outpath + "data/2/" + file + ".pkl", "wb"))
        
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

def calculate_spectograms_inner(data, window_width, fs,
                                f_min, f_max, f_window_width, f_window_step_sz,
                                start_idx, filter_step_width = 1, peak_spread = 1.5, peak_thresh = 1.1):
    
    nfft = int(fs*4)
    
    print("Processing idx %i of %i" % (start_idx, len(data) - window_width))
    data = np.asarray(data)
    loc_hf_data = ff.fir(np.copy(data), 300, None, filter_step_width, fs)
    (loc_burst_hf_data, loc_non_burst_hf_data) = preprocess_data(loc_hf_data, fs, peak_spread, peak_thresh)

    (bins, loc_lf_psd) = scipy.signal.welch(data, fs, window = "hann", nperseg = nfft, noverlap = int(nfft/2), nfft = nfft, detrend = False, return_onesided = True)
    (_, loc_burst_hf_psd) = scipy.signal.welch(loc_burst_hf_data, fs, window = "hann", nperseg = nfft, noverlap = int(nfft/2), nfft = nfft, detrend = False, return_onesided = True)
    (_, loc_non_burst_hf_psd) = scipy.signal.welch(loc_non_burst_hf_data, fs, window = "hann", nperseg = nfft, noverlap = int(nfft/2), nfft = nfft, detrend = False, return_onesided = True)
    min_f_bin_idx = np.argmin(np.abs(bins - f_min))
    max_f_bin_idx = np.argmin(np.abs(bins - f_max))
    
    loc_burst_dmi_scores = list()
    loc_non_burst_dmi_scores = list()
    for f_idx in np.arange(f_min, f_max, f_window_step_sz):
        loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)        
        if ((loc_burst_hf_data != 0).all()):
            (loc_dmi_score, _, _) = dmi.run(loc_dmi_lf_data, loc_burst_hf_data, 10, 1)
        else:
            loc_dmi_score = 0 #There cannot be any pack if there is no data in this segment
        loc_burst_dmi_scores.append(loc_dmi_score)
        
        loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)
        if ((loc_non_burst_hf_data != 0).all()):
            (loc_dmi_score, _, _) = dmi.run(loc_dmi_lf_data, loc_non_burst_hf_data, 10, 1)
        else:
            loc_dmi_score = 0 #There cannot be any pack if there is no data in this segment
        loc_non_burst_dmi_scores.append(loc_dmi_score)
            
    return (loc_lf_psd[min_f_bin_idx:max_f_bin_idx], loc_burst_hf_psd[min_f_bin_idx:max_f_bin_idx], loc_non_burst_hf_psd[min_f_bin_idx:max_f_bin_idx],
            loc_burst_dmi_scores, loc_non_burst_dmi_scores)

import skimage.morphology

def calculate_spectograms(data, fs, peak_spread, peak_thresh, filter_step_width = 1,
                          lf_zoom_factor = 1, lf_offset_factor = 0, hf_zoom_factor = 1, hf_offset_factor = 0, pac_thresh = 0.6,
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (overwrite or os.path.exists(outpath + "data/3/" + file + ".pkl") == False):
        window_width = fs*4
        window_step_sz = int(fs/2)
        
        f_min = 2
        f_max = 10
        f_window_width = 0
        f_window_step_sz = 0.25
            
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
    
    a = 20
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
     pac_burst_specific_strength, pac_non_burst_specific_strength) = score_pac(lf_psd, hf_burst_psd, hf_non_burst_psd,
                                                                               dmi_burst_scores, dmi_non_burst_scores,
                                                                               2, 2, -2,
                                                                               3, pac_thresh)
     
    #----------------------- pac_burst_strength = 0; pac_non_burst_strength = 0;
    #----------------- pac_burst_specificity = 0; pac_non_burst_specificity = 0;
    #----- pac_burst_specific_strength = 0; pac_non_burst_specific_strength = 0;
        
    if (visualize):
        
        dmi_burst_scores[dmi_burst_scores < float(pac_thresh)] = 0
        dmi_non_burst_scores[dmi_non_burst_scores < float(pac_thresh)] = 0
        
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
            ax.set_yticks(np.asarray([0, 4, 8, 12, 16, 20, 24, 28])+3)
            ax.set_yticklabels(([2, 3, 4, 5, 6, 7, 8, 9][::-1]))
        
        fig.set_tight_layout(tight = True)
        
        fig.savefig(outpath + "/img/3/" + file + image_format)
        
    return (pac_burst_strength, pac_non_burst_strength,
            pac_burst_specificity, pac_non_burst_specificity, 
            pac_burst_specific_strength, pac_non_burst_specific_strength)

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
    dmi_burst_scores_enlarged[dmi_burst_scores_enlarged < float(pac_thresh)] = 0
    dmi_non_burst_scores_enlarged[dmi_non_burst_scores_enlarged < float(pac_thresh)] = 0

    dmi_burst_scores_sng = np.sign(np.multiply(hf_burst_psd_fused, dmi_burst_scores_enlarged))
    dmi_non_burst_scores_sgn = np.sign(np.multiply(hf_non_burst_psd_fused, dmi_non_burst_scores_enlarged))
    
    pac_burst_specificity = np.sum(dmi_burst_scores_sng[dmi_burst_scores_sng > 0])/np.sum(np.abs(dmi_burst_scores_sng)) if ((dmi_burst_scores_sng != 0).any()) else 0
    pac_non_burst_specificity = np.sum(dmi_non_burst_scores_sgn[dmi_non_burst_scores_sgn > 0])/np.sum(np.abs(dmi_non_burst_scores_sgn)) if ((dmi_non_burst_scores_sgn != 0).any()) else 0
    
    pac_burst_strength = np.sum(dmi_burst_scores)/dmi_burst_scores.shape[1]
    pac_non_burst_strength = np.sum(dmi_non_burst_scores)/dmi_non_burst_scores.shape[1]
    
    tmp = np.zeros(dmi_burst_scores_sng.shape); tmp[dmi_burst_scores_sng > 0] = 1; pac_burst_specific_strength = np.sum(np.multiply(tmp, dmi_burst_scores))/dmi_burst_scores.shape[1]
    tmp = np.zeros(dmi_non_burst_scores_sgn.shape); tmp[dmi_non_burst_scores_sgn > 0] = 1; pac_non_burst_specific_strength = np.sum(np.multiply(tmp, dmi_non_burst_scores))/dmi_non_burst_scores.shape[1]
    
    return (pac_burst_strength, pac_non_burst_strength, 
            pac_burst_specificity, pac_non_burst_specificity, 
            pac_burst_specific_strength, pac_non_burst_specific_strength)

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
        
def main(mode = "power", overwrite = False, visualize = False):
    meta_file = methods.data_io.ods.ods_data("../../../../data/meta.ods")
    meta_data = meta_file.get_sheet_as_dict("tremor")
    in_path = "../../../../data/tremor/data_for_python/"
    out_path = "../../../../results/tremor/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        
        if (file == ""):
            continue
        
        #------------------------------------------- if (file != "633-s1-1194"):
            #---------------------------------------------------------- continue
        
        if (meta_data["valid_data"][file_idx] == 0):
            continue
        
        print(file)
        
        if (overwrite == True):
            file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
            file_data = pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))
            
            fs_data01 = int(file_hdr[20]['fs'])       
            loc_data = ff.fir(np.asarray(file_data[20]), 1, None, 0.1, fs_data01)
        else:
            file_hdr = None
            file_data = None
            fs_data01 = None
            loc_data = None
                    
        lf_f_min = meta_data["lf f min"][file_idx]
        lf_f_max = meta_data["lf f max"][file_idx]
        hf_f_min = meta_data["hf f min"][file_idx]
        hf_f_max = meta_data["hf f max"][file_idx]            
      
        if ("power" in mode):
            tremor_values = plot_hf_lf_components(loc_data, fs_data01, lf_f_min, lf_f_max, hf_f_min, hf_f_max, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], outpath = out_path, file = file,
                                                overwrite = overwrite, visualize = visualize)
            meta_data["tremor lfp strength 1"][file_idx] = float(tremor_values[0])
            meta_data["tremor overall strength 1"][file_idx] = float(tremor_values[1])
            meta_data["tremor burst strength 1"][file_idx] = float(tremor_values[2])
            meta_data["tremor non burst strength 1"][file_idx] = float(tremor_values[3])
                        
        if ("overall pac" in mode):
            pac_values = calculate_dmi(loc_data, fs_data01, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], outpath = out_path, file = file,
                                       overwrite = overwrite, visualize = visualize)
            meta_data["pac overall strength 2"][file_idx] = float(pac_values[0])
            meta_data["pac burst strength 2"][file_idx] = float(pac_values[1])
            meta_data["pac non burst strength 2"][file_idx] = float(pac_values[2])
            
        if("specific pac" in mode):
            pac_values = calculate_spectograms(loc_data, fs_data01, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], 
                                               lf_zoom_factor = meta_data["lf_zoom_factor"][file_idx], lf_offset_factor = meta_data["lf_offset_factor"][file_idx],
                                               hf_zoom_factor = meta_data["hf_zoom_factor"][file_idx], hf_offset_factor = meta_data["hf_offset_factor"][file_idx], pac_thresh = meta_data["pac_thresh"][file_idx],
                                               outpath = out_path, file = file, overwrite = overwrite, visualize = visualize)
            meta_data["pac burst strength 3"][file_idx] = float(pac_values[0])
            meta_data["pac non burst strength 3"][file_idx] = float(pac_values[1])
            meta_data["pac burst specificity 3"][file_idx] = float(pac_values[2])
            meta_data["pac non burst specificity 3"][file_idx] = float(pac_values[3])
            meta_data["pac burst specific strength 3"][file_idx] = float(pac_values[4])
            meta_data["pac non burst specific strength 3"][file_idx] = float(pac_values[5])
        
        if (len(mode) == 3):
            meta_file.modify_sheet_from_dict("tremor", meta_data)
            meta_file.write_file()
        
        
        plt.close("all")
    
    if (len(mode) != 3):
        meta_file.modify_sheet_from_dict("tremor", meta_data)
        meta_file.write_file()
    print("Terminated successfully")
    
#main(["power", "overall pac", "specific pac"], overwrite = False, visualize = True)
main(["specific pac"], overwrite = False, visualize = True)
#main(["power"], overwrite = True, visualize = True)


