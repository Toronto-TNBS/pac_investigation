'''
Created on Feb 23, 2022

@author: voodoocode
'''

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import misc.spike2_conv as conv
import numpy as np

import pyexcel_ods
import scipy.signal

import finn.filters.frequency as ff
import finn.cfc.pac as pac

import finn.misc.timed_pool as tp
import methods.detection.bursts

def rm_stim_artifact(data, fs, trans_width = 1, fft_win_sz = 1, pad_type = "zero", 
                     ripple_pass_band = 10e-5, stop_band_suppression = 10e-7):
    nyq = fs/2

    peaks = scipy.signal.find_peaks(scipy.signal.welch(data, fs, window = "hanning", nperseg = fs, noverlap = fs // 2, nfft = fs, detrend = None)[1], distance = 80)[0]
    peaks = np.delete(peaks, np.argwhere(peaks > 4500).squeeze())

    freq = [0]; gain = [1]
    for peak in peaks:
        if (peak < 80):
            continue
        
        freq.extend([peak - 3, peak - 2, peak + 2, peak + 3])
        gain.extend([       1,        0,        0,        1])

    freq.append(nyq); gain.append(1)

    freq = np.asarray(freq, dtype = float)
    freq /= nyq
    
    N = ff.__estimate_FIR_coeff_num(fs, trans_width, ripple_pass_band, stop_band_suppression) + 1
    coeffs = scipy.signal.firwin2(numtaps = N, freq = freq, gain = gain, window = 'hamming') # Guarantees a linear phase
    data = np.asarray(data, dtype = np.float32)
    coeffs = np.asarray(coeffs, dtype = np.float32)
    data = ff.__overlap_add(data, coeffs, fs, trans_width, fft_win_sz, pad_type) # Removes shift introduced by the filter
    
    data = ff.fir(data, None, 4000, 1, fs)
                 
    return data

def screen():
    meta = pyexcel_ods.get_data(open(path + "meta.ods", "rb"))["Sheet1"]

    for (file_idx, file_meta_info) in enumerate(meta):
        if (file_idx == 0):
            continue
        
        f_name_idx          = meta[0].index("file")
        data_channel_idx    = meta[0].index("data_channel")
        f_min_idx           = meta[0].index("lfp_f_min")
        f_max_idx           = meta[0].index("lfp_f_max")
        valid_idx           = meta[0].index("valid")
        
        if (file_meta_info[valid_idx] == False):
            continue 

        file = file_meta_info[f_name_idx]

        print(file)
    
        file_path = path + file
        
        (_, ch_infos, data) = conv.read_file(file_path)
        
        fs = int(ch_infos[file_meta_info[data_channel_idx]]["fs"])
        raw_data = data[file_meta_info[data_channel_idx]]
        
        lfp_data = ff.fir(np.copy(raw_data), 2, 58, 1, fs, 10e-5, 10e-7, pad_type = "zero", mode = "fast")
         
        filt_raw_data = None
        if ("100Hz" in file):
            filt_raw_data = rm_stim_artifact(np.copy(raw_data), fs)
             
        my_dpi = 96
        (fig, axes) = plt.subplots(3, 2, figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
        fig.suptitle(file + " | " + str(fs))
        axes[0, 1].plot(scipy.signal.welch(lfp_data, fs, window = "hanning", nperseg = int(fs), noverlap = fs // 2, nfft = int(fs), detrend = "linear")[1][:50], linewidth = .3)
        axes[1, 1].plot(scipy.signal.welch(raw_data, fs, window = "hanning", nperseg = int(fs), noverlap = fs // 2, nfft = int(fs), detrend = "linear")[1][:1000], linewidth = .3)
        axes[2, 1].plot(scipy.signal.welch(raw_data, fs, window = "hanning", nperseg = int(fs), noverlap = fs // 2, nfft = int(fs), detrend = "linear")[1], linewidth = .3)
 
        axes[0, 0].plot(lfp_data, linewidth = .3)
        axes[1, 0].plot(raw_data, linewidth = .3)
        axes[2, 0].plot(raw_data, linewidth = .3)
         
        if (filt_raw_data is not None):
            plt.figure()
            plt.plot(raw_data)
            plt.plot(filt_raw_data)
         
        plt.tight_layout()
        plt.show(block = True)

def calculate_spectograms_inner(data, window_width, fs,
                                f_min, f_max, f_window_width, f_window_step_sz, start_idx, 
                                param_type):
    
    #print("Processing idx %i of %i" % (start_idx, len(data) - window_width))
    data = np.asarray(data)
    
    if (param_type == "tremor"):
        nfft = int(fs*4)
    elif(param_type == "beta"):
        nfft = int(fs)
    (bins, psd_lf) = scipy.signal.welch(ff.fir(np.copy(data), f_min, f_max, 1, fs), fs, window = "hann", nperseg = nfft, noverlap = int(nfft/2), nfft = nfft, detrend = False, return_onesided = True)
    (_,    psd_hf) = scipy.signal.welch(np.abs(scipy.signal.hilbert(ff.fir(np.copy(data),   300,  None, 1, fs))), fs, window = "hann", nperseg = nfft, noverlap = int(nfft/2), nfft = nfft, detrend = False, return_onesided = True)
    min_f_bin_idx = np.argmin(np.abs(bins - f_min))
    max_f_bin_idx = np.argmin(np.abs(bins - f_max))
    
    loc_hf_data = ff.fir(np.copy(data), 300, None, 1, fs)
    loc_hf_data = np.abs(scipy.signal.hilbert(loc_hf_data))
    pac_scores = list()
    for f_idx in np.arange(f_min, f_max, f_window_step_sz):
        loc_lf_data = ff.fir(np.copy(data), f_idx - f_window_width/2, f_idx + f_window_width/2, 1, fs)
        (pac_score, _, _) = pac.run_dmi(loc_lf_data, loc_hf_data, 10, 1)
        pac_scores.append(pac_score)
            
    return (psd_lf[min_f_bin_idx:max_f_bin_idx], psd_hf[min_f_bin_idx:max_f_bin_idx], pac_scores)

def compute_histograms(data, fs, param_type = "tremor"):
    
    thread_cnt = 30
    if (param_type == "tremor"):
        t_window_width = fs*4
        t_window_step_sz = int(fs/2)
        
        f_min = 2
        f_max = 10
        f_window_width = 0
        f_window_step_sz = 0.25
    elif(param_type == "beta"):
        t_window_width = fs
        t_window_step_sz = int(fs/2)
        
        f_min = 12
        f_max = 35
        f_window_width = 2
        f_window_step_sz = 1
    
    tmp = np.asarray(tp.run(thread_cnt, calculate_spectograms_inner,
                            [(data[start_idx:(start_idx + t_window_width)], t_window_width, fs,
                              f_min, f_max, f_window_width, f_window_step_sz,
                              start_idx, param_type) for start_idx in np.arange(0, len(data) - t_window_width, t_window_step_sz)],
                            max_time = None, delete_data = True))
    
    (psd_lf, psd_hf, pac_scores) = np.swapaxes(tmp, 0, 1)
    
    psd_lf = np.flip(psd_lf, axis = 1).transpose()
    psd_hf = np.flip(psd_hf, axis = 1).transpose()
    pac_scores = np.flip(pac_scores, axis = 1).transpose()
    
    a = 1.7
    psd_lf = (np.power(a, np.log(psd_lf)) - 1)/(a-1)
    psd_hf = (np.power(a, np.log(psd_hf)) - 1)/(a-1)
    #===========================================================================
    # a = 20
    # pac_scores = (np.power(a, pac_scores)-1)/(a - 1)
    #===========================================================================
    
    psd_lf = psd_lf - np.min(psd_lf); psd_lf = psd_lf / np.max(psd_lf)
    psd_lf = (psd_lf - 0.5)*2
    psd_hf = psd_hf - np.min(psd_hf); psd_hf = psd_hf / np.max(psd_hf)
    psd_hf = (psd_hf - 0.5)*2
    
    psd_thresh = 0.1
    psd_lf[psd_lf < float(psd_thresh)] = -1
    psd_hf[psd_hf < float(psd_thresh)] = -1
    
    pac_thresh = 0.6
    pac_scores[pac_scores < float(pac_thresh)] = 0
    
    return (psd_lf, psd_hf, pac_scores)

def evaluate(path = "/mnt/data/Professional/UHN/projects/old/pac_investigation/data/demo/"):
    meta = pyexcel_ods.get_data(open(path + "meta.ods", "rb"))["Sheet1"]

    for (file_idx, file_meta_info) in enumerate(meta):
        if (file_idx == 0):
            continue
        
        f_name_idx          = meta[0].index("file")
        data_channel_idx    = meta[0].index("data_channel")
        acc_channel_idx     = meta[0].index("acc_channel")
        f_min_idx           = meta[0].index("lfp_f_min")
        f_max_idx           = meta[0].index("lfp_f_max")
        param_type_idx      = meta[0].index("type")
        t_min_idx           = meta[0].index("t_min")
        t_max_idx           = meta[0].index("t_max")
        valid_idx           = meta[0].index("valid")
        
        if (file_meta_info[valid_idx] == False):
            continue 
        
        file = file_meta_info[f_name_idx]
        file_path = path + file
        print(file)
        
        #=======================================================================
        # if (file != "STN56-100Hz-long.smr"):
        #     continue
        #=======================================================================
        
        #=======================================================================
        # if (file != "658-TREMOR-WITH-ACCELEROMETER-AND-MOV-DESYNCH.smr"):
        #     continue
        #=======================================================================
        
        #=======================================================================
        # if (file != "658-TREMOR-WITH-ACCELEROMETER-AND-MOV-DESYNCH.smr"):
        #     continue
        # 
        # file = "639-2376.smr"
        # path = "/mnt/data/Professional/UHN/projects/old/pac_investigation/code/beta_pac/analysis/other/"
        # file_path = path + file
        #=======================================================================
        
        (file_info, ch_infos, data) = conv.read_file(file_path)
        
        acc_fs = int(ch_infos[file_meta_info[acc_channel_idx]]["fs"])
        fs = int(ch_infos[file_meta_info[data_channel_idx]]["fs"])
        acc_data = data[file_meta_info[acc_channel_idx]][int(acc_fs*file_meta_info[t_min_idx]):int(acc_fs*file_meta_info[t_max_idx])]
        data = data[file_meta_info[data_channel_idx]][int(fs*file_meta_info[t_min_idx]):int(fs*file_meta_info[t_max_idx])]
        if ("100Hz" in file):
            data = rm_stim_artifact(np.copy(data), fs)
        param_type = file_meta_info[param_type_idx]
        
        (psd_lf, psd_hf, pac_scores) = compute_histograms(data, fs, param_type)
        
        (fig, axes) = plt.subplots(5, 1)
        axes[0].plot(acc_data)
        axes[1].plot(data)
        axes[2].imshow(psd_lf,     vmin = -1, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        axes[3].imshow(psd_hf,     vmin = -1, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        axes[4].imshow(pac_scores, vmin =  0, vmax = 1, cmap='seismic', aspect = 'auto', interpolation = 'lanczos')
        
        for (ax_idx, ax) in enumerate(axes.reshape(-1)):
            if (ax_idx == 0 or ax_idx == 1):
                continue
            if (param_type == "tremor"):
                ax.set_yticks(np.asarray([0, 4, 8, 12, 16, 20, 24, 28])+3)
                ax.set_yticklabels(([2, 3, 4, 5, 6, 7, 8, 9][::-1]))
            elif (param_type == "beta"):
                ax.set_yticks([0, 5, 10, 15, 20])
                ax.set_yticklabels(([35, 30, 25, 20, 15]))
    
        fig.set_tight_layout(tight = True)
        fig.savefig(file+".svg")

path = "/mnt/data/Professional/UHN/projects/old/pac_investigation/data/demo/"
#screen()
evaluate()

print("terminated")



