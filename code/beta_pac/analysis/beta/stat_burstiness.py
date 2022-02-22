'''
Created on Dec 7, 2021

@author: voodoocode
'''

import methods.detection.bursts
import methods.data_io.ods

import os

import pickle
import numpy as np
import scipy.signal

import finn.filters.frequency as ff
import finn.statistical.glmm as glmm

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

def get_data(stat_data, data_type):
    meta_file = methods.data_io.ods.ods_data("../../../../data/meta.ods")
    if (data_type == 0):
        meta_data = meta_file.get_sheet_as_dict("beta")
        in_path = "../../../../data/beta/data_for_python/"
    elif(data_type == 1):
        meta_data = meta_file.get_sheet_as_dict("tremor")
        in_path = "../../../../data/tremor/data_for_python/"
    for (file_idx, file) in enumerate(meta_data["file"]):
        if (file == ""):
            continue
        if (int(meta_data["valid_data"][file_idx]) == 0):
                continue
        
        print("%2.1f of %i" % (file_idx/len(meta_data["file"])*100, 100), file)
            
        file_hdr = pickle.load(open(in_path+file+".txt_conv_hdr.pkl", "rb"))
        file_data = pickle.load(open(in_path+file+".txt_conv_data.pkl", "rb"))
        fs = int(file_hdr[20]['fs'])
        loc_data = np.asarray(file_data[20])
        loc_data = ff.fir(loc_data, 2, None, 0.1, fs)
        
        peak_spread = float(meta_data["peak_spread"][file_idx])
        peak_thresh = float(meta_data["peak_thresh"][file_idx])
        
        binarized_data = methods.detection.bursts.identify_peaks(np.copy(loc_data), fs, 300, None, peak_spread, peak_thresh, "negative", "auto")
        
        spike_data = ff.fir(np.copy(loc_data), 300, None, 1, fs)
        spike_data[spike_data > 0] = 0
        buffer_area = int(fs/300)
        spikes = np.zeros(binarized_data.shape)
        (spike_loc, _) = scipy.signal.find_peaks(np.abs(spike_data), height = peak_thresh, distance = buffer_area)
        spikes[spike_loc] = 1
        
        ratio_of_spikes_within_bursts = np.sum(spikes[np.argwhere(binarized_data == 1)])/(np.sum(spikes[np.argwhere(binarized_data == 1)]) + np.sum(spikes[np.argwhere(binarized_data == -1)]))
        firing_rate = np.sum(spikes) / (len(binarized_data) / fs)
        
        loc_data = [ratio_of_spikes_within_bursts, firing_rate, meta_data["patient_id"][file_idx], meta_data["trial"][file_idx], meta_data["hf auto"][file_idx], data_type]
        
        stat_data.append(loc_data)
        
        #---------------------------------------------- plt.plot(binarized_data)
        #------------------------------------------------------ plt.plot(spikes)
        #------------------------------------------------ plt.show(block = True)
    
    print("A")

def main(overwrite = False):
    if (overwrite == True or os.path.exists("stat_data.pkl") == False):
        stat_data = list()
        get_data(stat_data, 0)#beta
        get_data(stat_data, 1)#tremor
        stat_data = np.asarray(stat_data)
        pickle.dump(stat_data, open("stat_data.pkl", "wb"))
    else:
        stat_data = pickle.load(open("stat_data.pkl", "rb"))
    stat_data = np.asarray(stat_data, dtype = float)
    
    factor_names = ["spike_ratio", "firing_rate", "patient", "trial", "peak_type", "data_type"]
    factor_types = ["continuous", "continuous", "categorical", "categorical", "categorical", "categorical"]
    contrasts = "list(spike_ratio = contr.sum, firing_rate = contr.sum, patient = contr.sum, trial = contr.sum, peak_type = contr.sum, data_type = contr.sum)"
    
    formula = "spike_ratio ~ peak_type + (1|patient) + (1|trial)"
    loc_data = np.copy(stat_data)
    loc_data = loc_data[np.argwhere(loc_data[:, 0] != 0).squeeze(), :]
    loc_data = loc_data[np.argwhere(loc_data[:, 5] == 0).squeeze(), :] # beta only
    stats = glmm.run(loc_data, factor_names, factor_types, formula, contrasts, "gaussian")
    stats = np.asarray(stats)
    print(stats)
    print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))

    formula = "spike_ratio ~ peak_type + (1|patient) + (1|trial)"
    loc_data = np.copy(stat_data)
    #loc_data[np.argwhere(loc_data[:, 0] != 0).squeeze(), 0] = 1
    loc_data = loc_data[np.argwhere(loc_data[:, 5] == 0).squeeze(), :] # beta only
    stats = glmm.run(loc_data, factor_names, factor_types, formula, contrasts, "gaussian")
    stats = np.asarray(stats)
    print(stats)
    print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))

    formula = "firing_rate ~ peak_type + (1|patient) + (1|trial)"
    loc_data = np.copy(stat_data)
    loc_data = loc_data[np.argwhere(loc_data[:, 5] == 0).squeeze(), :] # beta only
    stats = glmm.run(loc_data, factor_names, factor_types, formula, contrasts, "gaussian")
    stats = np.asarray(stats)
    print(stats)
    print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))

    print("\n\n")
    formula = "spike_ratio ~ data_type + (1|patient) + (1|trial)"
    loc_data = np.copy(stat_data)
    loc_data = loc_data[np.argwhere(loc_data[:, 0] != 0).squeeze(), :]
    loc_data = loc_data[np.argwhere(loc_data[:, 4] == 1).squeeze(), :] # strong only
    stats = glmm.run(loc_data, factor_names, factor_types, formula, contrasts, "gaussian")
    stats = np.asarray(stats)
    print(stats)
    print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))


main(overwrite = False)




