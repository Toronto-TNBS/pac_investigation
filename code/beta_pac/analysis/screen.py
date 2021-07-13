# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: VoodooCode
"""

import pickle
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import methods.data_io.ods
    
import scipy.signal
import numpy as np

import finn.filters.frequency as ff
        
import os

import methods.detection.bursts

def main(mode = "beta"):
    meta = methods.data_io.ods.ods_data("../../../data/meta.ods")
    meta_data = meta.get_sheet_as_dict(mode)
    
    in_path = "../../../data/"+mode+"/data_for_python/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        
        print(file)
        
        if (file == ""):
            continue
        
        #-------------------------------------- if (file != "2781_s2_147-BETA"):
            #---------------------------------------------------------- continue
        
        #---------------------- if (meta_data["height-checked"][file_idx] == 1):
            #---------------------------------------------------------- continue
        
        print(file)
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        
        file_data = ff.fir(file_data, 300, None, 1, int(file_hdr[20]['fs']))
        
        #--- file_data = ff.fir(file_data, 70, None, 1, int(file_hdr[20]['fs']))
        #--- file_data = ff.fir(file_data, 135, 125, 1, int(file_hdr[20]['fs']))
        #--- file_data = ff.fir(file_data, 205, 195, 1, int(file_hdr[20]['fs']))
        #--- file_data = ff.fir(file_data, 405, 395, 1, int(file_hdr[20]['fs']))
        
        file_data_mod = np.copy(file_data)
        file_data_mod[file_data > 0] = 0
        height = meta_data["peak_thresh"][file_idx] if (meta_data["peak_thresh"][file_idx] != "") else 0
        (peaks, _) = scipy.signal.find_peaks(np.abs(file_data_mod), height = float(height))
        
        data2 = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        print(len(peaks), len(peaks)/len(data2) * int(file_hdr[20]['fs']))
        
        
        (fig, axes) = plt.subplots(2, 1)
        axes[0].plot(file_data, zorder = 0)
        axes[0].scatter(peaks, np.asarray(file_data)[peaks], color = "red", zorder = 1)
        axes[0].plot([0, len(file_data)], [float(meta_data["peak_thresh"][file_idx]), float(meta_data["peak_thresh"][file_idx])],
                     color = "black", zorder = 2)
        axes[0].plot([0, len(file_data)], [-float(meta_data["peak_thresh"][file_idx]), -float(meta_data["peak_thresh"][file_idx])],
                     color = "black", zorder = 2)
        axes[1].psd(file_data, NFFT = int(file_hdr[20]['fs']), Fs = int(file_hdr[20]['fs']))
        fig.suptitle(file + ": " + str(height))
        
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50,100,640, 545)
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(0, 0, 950, 1080)
        
        fs = file_hdr[20]['fs']
        data = ff.fir(file_data, 300, None, 1, fs)
        data = np.copy(data)
        data[data > 0] = 0
        
        (peaks, _) = scipy.signal.find_peaks(np.abs(data), height = float(meta_data["peak_thresh"][file_idx]))
        peak_data = np.zeros(data.shape)
        peak_data[peaks] = 1
        binarized_data = methods.detection.bursts.identify_peaks(ff.fir(np.copy(np.asarray(data)), 300, None, 1, fs), fs, 70, None, float(meta_data["peak_spread"][file_idx]), float(meta_data["peak_thresh"][file_idx]))
        
        burst_data = peak_data[np.argwhere(binarized_data == 1).squeeze()]
        non_burst_data = peak_data[np.argwhere(binarized_data == -1).squeeze()]
        
        burst_spikes_percentage = np.sum(burst_data) / np.sum(peak_data) if (len(peak_data) != 0) else -1
        non_burst_spikes_percentage = np.sum(non_burst_data) / np.sum(peak_data) if (len(peak_data) != 0) else -1
        
        spikes_per_second = np.sum(peak_data) / len(peak_data) * fs if (len(peak_data) != 0) else -1
        burst_spikes_per_second = np.sum(burst_data)/len(burst_data) * fs if (len(burst_data) != 0) else -1
        non_burst_spikes_per_second = np.sum(non_burst_data)/len(non_burst_data) * fs if (len(non_burst_data) != 0) else -1

        print((burst_spikes_percentage, non_burst_spikes_percentage, burst_spikes_per_second, non_burst_spikes_per_second, spikes_per_second))

        plt.show(block = True)
        
    print("Terminated successfully")
    
main()


