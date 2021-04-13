# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: Luka
"""

import pickle
import matplotlib.pyplot as plt

import methods.data_io.read_ods
    
import scipy.signal
import numpy as np

import finn.filters.frequency as ff
        
def main():
    
    meta_data = methods.data_io.read_ods.read_file("../../../../data/meta.ods", "beta")
    in_path = "../../../../data/data_for_python/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        print(meta_data["height-checked"][file_idx])
        
        if (meta_data["height-checked"][file_idx] == 1):
            continue
        
        print(file)
        
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = np.asarray(pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))[20])
        file_data = ff.fir(file_data, 70, None, 1, int(file_hdr[20]['fs']))  
        file_data_mod = np.copy(file_data)
        file_data_mod[file_data > 0] = 0
        height = meta_data["peak_thresh"][file_idx]
        (peaks, _) = scipy.signal.find_peaks(np.abs(file_data_mod), height = height)
        
        plt.plot(file_data)
        plt.scatter(peaks, np.asarray(file_data)[peaks], color = "red")
        plt.title(file + ": " + str(height))
        plt.show(block = True)
        
    print("Terminated successfully")
    
main()


