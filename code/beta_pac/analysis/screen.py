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
    
    all_dist = list()
    for (file_idx, file) in enumerate(meta_data["file"]):
                
        if (file == ""):
            continue
        
        if (int(meta_data["valid_data"][file_idx]) == 0):
            continue
        
        #----------------------------------- if ("3304_tbd_s1_138" not in file):
            #---------------------------------------------------------- continue
        
        print(file)
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        
        file_data = ff.fir(file_data, 300, None, 1, int(file_hdr[20]['fs']))
        
        file_data_mod = np.copy(file_data)
        file_data_mod[file_data > 0] = 0
        height = meta_data["peak_thresh"][file_idx] if (meta_data["peak_thresh"][file_idx] != "") else 0
        (peaks, _) = scipy.signal.find_peaks(np.abs(file_data_mod), height = float(height), distance = int(int(file_hdr[20]['fs'])/1000/2))
        binarized_data = methods.detection.bursts.identify_peaks(np.copy(file_data), int(file_hdr[20]['fs']), 300, None, float(meta_data["peak_spread"][file_idx]), float(meta_data["peak_thresh"][file_idx]), "negative", "auto")
        burst_data = np.zeros(file_data.shape); burst_data[np.argwhere(binarized_data == 1)] = file_data[np.argwhere(binarized_data == 1)]
        non_burst_data = np.zeros(file_data.shape); non_burst_data[np.argwhere(binarized_data == -1)] = file_data[np.argwhere(binarized_data == -1)]
        
        data2 = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        print(len(peaks), len(peaks)/len(data2) * int(file_hdr[20]['fs']))
        
        fig = plt.figure(figsize = (8, 6))
        axes0 = plt.subplot(2, 2, 4)
        axes1 = plt.subplot(2, 2, 3)
        axes2 = plt.subplot(2, 1, 1)
        axes2.plot(file_data, zorder = 0)
        axes2.plot(burst_data, color = "#1a21b0", zorder = 1)
        axes2.plot(non_burst_data, color = "#5e62b8", zorder = 1)
        axes2.scatter(peaks, np.asarray(file_data)[peaks], color = "red", zorder = 2)
        axes2.plot([0, len(file_data)], [float(meta_data["peak_thresh"][file_idx]), float(meta_data["peak_thresh"][file_idx])],
                     color = "black", zorder = 3)
        axes2.plot([0, len(file_data)], [-float(meta_data["peak_thresh"][file_idx]), -float(meta_data["peak_thresh"][file_idx])],
                     color = "black", zorder = 3)
        axes2.set_xticks([int(file_hdr[20]['fs']) * x for x in range(int(len(file_data)/int(file_hdr[20]['fs']))+ 1)])
                
        distances = list()
        for (peak_idx, _) in enumerate(peaks[:-1]):
            distances.append(peaks[peak_idx + 1] - peaks[peak_idx])
        all_dist.extend(distances)

        
        if (mode == "tremor"):
            axes1.axvline(np.average(distances), color = "orange", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/.5), color = "green", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/1.5), color = "red", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/float(meta_data["peak_spread"][file_idx])), color = "black", lw = 2, zorder = 0)
        elif(mode == "beta"):
            axes1.axvline(np.average(distances), color = "orange", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/.5), color = "red", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/1.5), color = "green", lw = 5, zorder = 0)
            axes1.axvline(np.average(distances)*(1/float(meta_data["peak_spread"][file_idx])), color = "black", lw = 2, zorder = 0)
        axes1.hist(distances, range = [0, int(file_hdr[20]['fs'])/10*3], bins = 70, zorder = 1)
        axes1.set_xticks(np.arange(0, (int(file_hdr[20]['fs'])/10*3), np.average(distances)))
        axes1.set_xticklabels(np.arange(0, (int(file_hdr[20]['fs'])/10*3)/np.average(distances), 1, dtype = int))
        
        (bins, freq) = scipy.signal.welch(np.abs(scipy.signal.hilbert(file_data)), int(file_hdr[20]['fs']), "hanning", int(file_hdr[20]['fs']), int(float(file_hdr[20]['fs'])/2), int(file_hdr[20]['fs']), False, True, "density")
        axes0.plot(bins[:100], np.log(freq[:100]))
        
        fig.suptitle(file + ": " + str(height) + " | %2.2f" % (float(len(peaks)/(len(file_data)/file_hdr[20]['fs'])), ))
        
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50,100,640, 545)
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(0, 0, 950, 1080)
        mngr.window.setGeometry(0, 100, 1920, 1080-100)
        
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
    
    plt.figure()
    plt.hist(all_dist, range = [0, int(file_hdr[20]['fs'])/10*3], bins = 70)
    if (mode == "tremor"):
        plt.axvline(np.average(distances), color = "orange")
        plt.axvline(np.average(distances)*(1/.5), color = "green")
        plt.axvline(np.average(distances)*(1/1.5), color = "red")
    elif(mode == "beta"):
        plt.axvline(np.average(distances), color = "orange")
        plt.axvline(np.average(distances)*(1/.5), color = "red")
        plt.axvline(np.average(distances)*(1/1.5), color = "green")
    plt.show(block = True)
        
    print("Terminated successfully")
    
main()


