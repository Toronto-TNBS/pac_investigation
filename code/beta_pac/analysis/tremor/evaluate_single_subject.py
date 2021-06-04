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

import finn.misc.timed_pool as tp

import methods.detection.bursts
import methods.data_io.ods

import os

thread_cnt = 4

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

def plot_hf_lf_components(data, fs_data01, filt_low, filt_high, f_min = 2, f_max = 10, peak_spread = 1.5, peak_thresh = 1.1, 
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (overwrite or os.path.exists(outpath + "data/1/" + file + ".pkl") == False):
        # filtering
        lpf_data01 = ff.fir(np.asarray(data), None, 50, 0.1, fs_data01)
        bpf_data01 = ff.fir(np.asarray(data), filt_low, filt_high, 0.1, fs_data01)
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
        
        axes[0, 0].set_title("default: time domain")
        axes[1, 0].set_title("burst: time domain")
        axes[2, 0].set_title("non burst: time domain")
        axes[0, 1].set_title("default: frequency domain")
        axes[1, 1].set_title("burst: frequency domain")
        axes[2, 1].set_title("non burst: frequency domain")
        
        axes[1, 1].set_yticks([0, vmax])
        axes[2, 1].set_yticks([0, vmax])
        
        fig.savefig(outpath + "img/1/" + file + ".png")

def calculate_dmi(data, fs_data01, filt_low, filt_high, peak_spread, peak_thresh,
                  visualize = True, outpath = None, file = None, overwrite = True):
    if (overwrite or os.path.exists(outpath + "data/2/" + file + ".pkl") == False):
    
        high_freq_component = ff.fir(np.asarray(data), 300, None, 1, fs_data01)
        low_freq_component = ff.fir(np.asarray(data), filt_low, filt_high, 0.1, fs_data01)
        
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
        fig.savefig(outpath + "img/2/" + file + ".png")

def calculate_spectograms_inner(data, window_width, fs,
                                filt_low, filt_high, f_min, f_max, f_window_width, f_window_step_sz,
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
        loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)
        
        if ((loc_burst_hf_data == 0).all()):
            loc_dmi_score = 0
        else:
            (loc_dmi_score, _, _) = dmi.run(loc_dmi_lf_data, loc_burst_hf_data, 10, 1)
        loc_burst_dmi_scores.append(loc_dmi_score)
        
        loc_dmi_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, filter_step_width, fs)
        (loc_dmi_score, _, _) = dmi.run(loc_dmi_lf_data, loc_non_burst_hf_data, 10, 1)
        loc_non_burst_dmi_scores.append(loc_dmi_score)
            
    return (loc_lf_psd[min_f_bin_idx:max_f_bin_idx], loc_burst_hf_psd[min_f_bin_idx:max_f_bin_idx], loc_non_burst_hf_psd[min_f_bin_idx:max_f_bin_idx],
            loc_burst_dmi_scores, loc_non_burst_dmi_scores)

def calculate_spectograms(data, fs, filt_low, filt_high, peak_spread, peak_thresh, filter_step_width = 1,
                          visualize = True, outpath = None, file = None, overwrite = True):
    
    if (overwrite or os.path.exists(outpath + "data/3/" + file + ".pkl") == False):
        window_width = fs
        window_step_sz = int(fs/2)
        
        f_min = 2
        f_max = 11
        f_window_width = 2
        f_window_step_sz = 1
            
        tmp = np.asarray(tp.run(thread_cnt, calculate_spectograms_inner,
                                [(data[start_idx:(start_idx + window_width)], window_width, fs,
                                  filt_low, filt_high, f_min, f_max, f_window_width, f_window_step_sz,
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
        
    if (visualize):
    
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
        
        vmin_lf = np.min([np.min(lf_psd),])
        vmax_lf = np.max([np.max(lf_psd),])
        vmin_hf = np.min([np.min(hf_burst_psd), np.min(hf_non_burst_psd)])
        vmax_hf = np.max([np.max(hf_burst_psd), np.max(hf_non_burst_psd)])
        
        (fig, axes) = plt.subplots(3, 2)
        axes[0, 0].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = vmin_lf, vmax = vmax_lf)
        axes[0, 1].imshow(lf_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = vmin_lf, vmax = vmax_lf)
        axes[1, 0].imshow(hf_burst_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = vmin_hf, vmax = vmax_hf)
        axes[1, 1].imshow(hf_non_burst_psd, cmap='seismic', aspect = 'auto', interpolation = 'lanczos', vmin = vmin_hf, vmax = vmax_hf)
        axes[2, 0].imshow(dmi_burst_scores, vmin = 0.75, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        axes[2, 1].imshow(dmi_non_burst_scores, vmin = 0.75, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        
        axes[0, 0].set_title("burst: low frequency psd")
        axes[1, 0].set_title("burst: high frequency psd")
        axes[2, 0].set_title("burst: PAC: low freq. - high freq")
        
        axes[0, 1].set_title("non burst: low frequency psd")
        axes[1, 1].set_title("non burst: high frequency psd")
        axes[2, 1].set_title("non burst: PAC: low freq. - high freq")
        
        for ax in axes.reshape(-1):
            ax.set_yticks([0,1, 2, 3, 4, 5, 6, 7, 8])
            ax.set_yticklabels(([2, 3, 4, 5, 6, 7, 8, 9,10][::-1]))
        
        fig.set_tight_layout(tight = True)
        
        fig.savefig(outpath + "/img/3/" + file + ".png")
        
def main(overwrite = True, mode = "tremor"):
    meta_data = methods.data_io.ods.read_file("../../../../data/meta.ods", mode)
    in_path = "../../../../data/"+mode+"/data_for_python/"
    out_path = "../../../../results/"+mode+"/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        #=======================================================================
        # if (file != "664-cell1-c-stim-on"):
        #     continue
        #=======================================================================
        
        #=======================================================================
        # if (file == "664-cell1-c-stim-on" or file == "2623-s4-280"):
        #     continue
        #=======================================================================
        
        print(file)
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))
        
        filt_low = meta_data["fmin"][file_idx]
        filt_high = meta_data["fmax"][file_idx]
    
        fs_data01 = int(file_hdr[20]['fs'])
         
        #plot_hf_lf_components(file_data[20], fs_data01, filt_low, filt_high, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], outpath = out_path, file = file, overwrite = overwrite)
        #calculate_dmi(file_data[20], fs_data01, filt_low, filt_high, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], outpath = out_path, file = file, overwrite = overwrite)
        calculate_spectograms(file_data[20], fs_data01, filt_low, filt_high, peak_spread = meta_data["peak_spread"][file_idx], peak_thresh = meta_data["peak_thresh"][file_idx], outpath = out_path, file = file, overwrite = overwrite)
        
        #plt.show(block = True)
        plt.close("all")
    print("Terminated successfully")
    
main()


