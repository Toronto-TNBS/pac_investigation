# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: VoodooCode
"""

import pickle
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import methods.data_io.read_ods
    
import scipy.signal
import numpy as np

import finn.filters.frequency as ff
        
import os        

def main(mode = "beta"):    
    meta_data = methods.data_io.read_ods.read_file("../../../data/meta.ods", mode)
    in_path = "../../../data/"+mode+"/data_for_python/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        
        if (meta_data["height-checked"][file_idx] == 1):
            continue
        
        print(file)
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        file_data = ff.fir(file_data, 70, None, 1, int(file_hdr[20]['fs']))  
        file_data = ff.fir(file_data, 135, 125, 1, int(file_hdr[20]['fs']))  
        file_data = ff.fir(file_data, 205, 195, 1, int(file_hdr[20]['fs']))  
        file_data = ff.fir(file_data, 405, 395, 1, int(file_hdr[20]['fs']))  
        file_data_mod = np.copy(file_data)
        file_data_mod[file_data > 0] = 0
        height = meta_data["peak_thresh"][file_idx] if (meta_data["peak_thresh"][file_idx] != "") else 0
        (peaks, _) = scipy.signal.find_peaks(np.abs(file_data_mod), height = height)
        
        (fig, axes) = plt.subplots(2, 1)
        axes[0].plot(file_data)
        axes[0].scatter(peaks, np.asarray(file_data)[peaks], color = "red")
        axes[1].psd(file_data, NFFT = int(file_hdr[20]['fs']), Fs = int(file_hdr[20]['fs']))
        fig.suptitle(file + ": " + str(height))
        
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50,100,640, 545)
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(0, 0, 950, 1080)


        plt.show(block = True)
        
    print("Terminated successfully")
    
main()


