'''
Created on Jul 12, 2021

@author: voodoocode
'''

import misc.spike2_conv as conv

import finn.filters.frequency as ff
import numpy as np

import sklearn.decomposition as sk_decomp
import matplotlib.pyplot as plt

import scipy.stats

def get_gradient(data):
    grad_data = np.zeros([data.shape[0], data.shape[1] - 1])
    
    for ch_idx in range(data.shape[0]):
        for samp_idx in range(data.shape[1] - 1):
            grad_data[ch_idx, samp_idx] = data[ch_idx, samp_idx] - data[ch_idx, samp_idx + 1]

    return grad_data

def quantify_segment_quality(data, min_spike_cnt = 100):
    SNR = list()
    for ch_idx in range(0, data.shape[0]):
        spike_value = np.mean(data[ch_idx, np.abs(scipy.stats.zscore(data[ch_idx, :])).argsort()[-min_spike_cnt:]])
        overall_value = np.mean(np.abs(scipy.stats.zscore(data[ch_idx, :]))) # noise-driven
        
        if (np.isinf((spike_value/overall_value))):
            print("A")
        
        SNR.append(spike_value/overall_value)
    
    return np.asarray(SNR)

def find_spikes(raw_data, fs, min_spike_cnt = 10, min_dist = 10):
    ff__data = list()
    for x in np.arange(1000, 5000, 500):
        ff__data.append(ff.fir(raw_data, x, None, 1, fs, 1e-5, 1e-7, pad_type = "zero", mode = "fast"))
    ff__data = np.asarray(ff__data)
    
    ff__data -= np.min(ff__data.reshape(-1))
    decomp = sk_decomp.NMF(ff__data.shape[0])
    decomp_data = decomp.fit_transform(ff__data.transpose()).transpose()
    seg_quality = quantify_segment_quality(decomp_data, min_spike_cnt)
    best_seg_idx = seg_quality.argsort()[-5:]
    
    for ch_idx in range(decomp_data.shape[0]):
        if (ch_idx in best_seg_idx):
            continue
        decomp_data[ch_idx, :] = 0
    
    refined_data = decomp.inverse_transform(decomp_data.transpose()).transpose()
    refined_data = np.sum(refined_data, axis = 0)
    
    spikes = np.argwhere(refined_data > (refined_data[refined_data.argsort()[-10]]/2)).squeeze()
    
    filt_spikes = [spikes[0]]
    for spike in spikes[1:]:
        if((spike - filt_spikes[-1]) < min_dist):
            if (refined_data[spike] > refined_data[filt_spikes[-1]]):
                filt_spikes[-1] = spike
        else:
            filt_spikes.append(spike)
    
    return filt_spikes
    
    return None

def main():
    file_path = "/home/voodoocode/Downloads/2891/2851_s1_635_power_pinch.smr"
    (file_info, ch_infos, data) = conv.read_file(file_path)
    fs = ch_infos[0]["fs"]
    print(fs)
    
    t_min = 0
    #t_max = int(fs * 0.1)
    t_max = int(fs * 1.1)
    
    ref_data = data[-1]#[t_min:t_max]
    raw_data = data[0]#[t_min:t_max]
    refined_data = ff.fir(raw_data, 1000, None, 10, fs, 1e-5, 1e-7, pad_type = "zero", mode = "fast")
                
    spikes = find_spikes(raw_data, fs, 6)
    
    plt.plot(raw_data, color = "orange")
    plt.plot(ff.fir(raw_data, 300, None, 1, fs, 1e-5, 1e-7, pad_type = "zero", mode = "fast"), color = "green")
    plt.scatter(spikes, raw_data[spikes])
    plt.plot(ref_data, linestyle = "--", color = "red")
    plt.show(block = True)
        
    plt.figure()
    plt.plot(raw_data, color = "orange")
    plt.plot(ff.fir(raw_data, 300, None, 1, fs, 1e-5, 1e-7, pad_type = "zero", mode = "fast"), color = "green")
    plt.plot(refined_data, color = "black")
    plt.scatter(np.argwhere(refined_data > (refined_data.argsort()[-10]/2)).squeeze(), refined_data[np.argwhere(refined_data > (refined_data.argsort()[-10]/2)).squeeze()], color = "black")
    #plt.plot(exp_data, color = "blue")
    plt.plot(ref_data, linestyle = "--", color = "red")
    
        
    plt.show(block = True)


main()
print("Terminated")