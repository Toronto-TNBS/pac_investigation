'''
Created on Oct 13, 2021

@author: voodoocode
'''

import numpy as np
import finn.filters.frequency as ff
import methods.detection.bursts

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import pickle

import methods.data_io.ods

def main(mode = "tremor"):
    meta = methods.data_io.ods.ods_data("../../../data/meta.ods")
    meta_data = meta.get_sheet_as_dict(mode)
    
    in_path = "../../../data/"+mode+"/data_for_python/"
    
    for (file_idx, file) in enumerate(meta_data["file"]):
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        data = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        fs = int(file_hdr[20]['fs'])
        peak_spread = float(meta_data["peak_spread"][file_idx])
        peak_thresh = float(meta_data["peak_thresh"][file_idx])
        
        if (float(meta_data["valid_data"][file_idx]) == 0):
            continue
        
        if (file == ""):
            continue
        
        #=======================================================================
        # if (file_idx < 83 - 2):
        #     continue
        #=======================================================================
                
        print(file, peak_thresh, peak_spread)
        #peak_spread = 1.5
        
        binarized_data = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh, buffer_area = "auto")
        
        
        binarized_data_old = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, peak_spread, peak_thresh, buffer_area = "auto")
        binarized_data_100 = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, 1.00, peak_thresh, buffer_area = "auto")
        binarized_data_125 = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, 1.25, peak_thresh, buffer_area = "auto")
        binarized_data_150 = methods.detection.bursts.identify_peaks(np.copy(np.asarray(data)), fs, 300, None, 1.50, peak_thresh, buffer_area = "auto")
        
        lpf_data = ff.fir(data, 5, 50, 0.1, fs, 10e-5, 10e-7, int(fs), "zero", "fast")
        hpf_data = ff.fir(data, 300, None, 0.1, fs, 10e-5, 10e-7, int(fs), "zero", "fast")
        
        start = int(0 * fs)
        end = len(data)# int(8 * fs)
        
        #burst_data = np.copy(hpf_data); burst_data[np.argwhere(binarized_data != 1).squeeze()] = np.nan
        burst_data_old = np.copy(hpf_data); burst_data_old[np.argwhere(binarized_data_old != 1).squeeze()] = np.nan
        burst_data_100 = np.copy(hpf_data); burst_data_100[np.argwhere(binarized_data_100 != 1).squeeze()] = np.nan
        burst_data_125 = np.copy(hpf_data); burst_data_125[np.argwhere(binarized_data_125 != 1).squeeze()] = np.nan
        burst_data_150 = np.copy(hpf_data); burst_data_150[np.argwhere(binarized_data_150 != 1).squeeze()] = np.nan
        #non_burst_data = np.copy(hpf_data); non_burst_data[np.argwhere(binarized_data != -1).squeeze()] = np.nan
        non_burst_data_old = np.copy(hpf_data); non_burst_data_old[np.argwhere(binarized_data_old != -1).squeeze()] = np.nan
        non_burst_data_100 = np.copy(hpf_data); non_burst_data_100[np.argwhere(binarized_data_100 != -1).squeeze()] = np.nan
        non_burst_data_125 = np.copy(hpf_data); non_burst_data_125[np.argwhere(binarized_data_125 != -1).squeeze()] = np.nan
        non_burst_data_150 = np.copy(hpf_data); non_burst_data_150[np.argwhere(binarized_data_150 != -1).squeeze()] = np.nan
        
        #=======================================================================
        # (fig, axes) = plt.subplots(2, 1)
        # fig.suptitle(str(file) + " | " + str(peak_thresh) + " | " + str(peak_spread))
        # #axes[0].plot(lpf_data[start:end], color = "black")
        # axes[0].plot(burst_data_100[start:end], color = "blue", zorder = 1, alpha = 0.33)
        # axes[0].plot(burst_data_125[start:end], color = "blue", zorder = 1, alpha = 0.33)
        # axes[0].plot(burst_data_150[start:end], color = "green", zorder = 1, alpha = 0.33)
        # axes[0].plot(hpf_data[start:end], color = "grey", zorder = 0)
        # 
        # axes[1].plot(non_burst_data_100[start:end], color = "orange", zorder = 1, alpha = 0.33)
        # axes[1].plot(non_burst_data_125[start:end], color = "blue", zorder = 1, alpha = 0.33)
        # axes[1].plot(non_burst_data_150[start:end], color = "green", zorder = 1, alpha = 0.33)
        # axes[1].plot(hpf_data[start:end], color = "grey", zorder = 0)
        #=======================================================================
        
        plt.figure()
        plt.title(str(file))
        plt.plot(burst_data_old[start:end], color = "red", zorder = 1)
        plt.plot(burst_data_100[start:end], color = "blue", zorder = 2)
        plt.plot(burst_data_125[start:end], color = "green", zorder = 3)
        plt.plot(burst_data_150[start:end], color = "orange", zorder = 4)
        plt.plot(hpf_data[start:end], color = "grey", zorder = 0)
        
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(0, 100, 1920, 1080/2)
        
        plt.show(block = True)
    
main()