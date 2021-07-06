'''
Created on Jun 10, 2021

@author: voodoocode
'''


import pickle
import numpy as np
import methods.data_io.ods

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.stats

buffer_width = 0
f_offset = 12

def calculate_variance(data, f_min, f_max):
    loc_data = np.copy(data)
    #loc_data -= np.min(loc_data); loc_data /= np.max(loc_data)
    loc_data = np.abs(scipy.stats.zscore(loc_data.reshape(-1)).reshape(loc_data.shape))
    
    return np.average(loc_data[:, (int(f_min) - f_offset):(int(f_max) - f_offset)].reshape(-1))

def get_values(path, subpath, mode):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    data_lf = list()
    data_hf_b = list()
    data_hf_nb = list()
    data_pac_b = list()
    data_pac_nb = list()
    for (f_idx, f_name) in enumerate(meta_info["file"]):

        #=======================================================================
        # if (f_name != "3304_tbd_s2_719"):
        #     continue
        #=======================================================================
        
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        if (int(meta_info["lf f min"][f_idx]) < buffer_width):
            continue
        if (int(meta_info["hf f min"][f_idx]) < buffer_width):
            continue
        
        values = [calculate_variance(data[0], meta_info["lf f min"][f_idx], meta_info["lf f max"][f_idx]), 
                  calculate_variance(data[1], meta_info["hf f min"][f_idx], meta_info["hf f max"][f_idx]),
                  calculate_variance(data[2], meta_info["hf f min"][f_idx], meta_info["hf f max"][f_idx]),
                  calculate_variance(data[3], meta_info["hf f min"][f_idx], meta_info["hf f max"][f_idx]),
                  calculate_variance(data[4], meta_info["hf f min"][f_idx], meta_info["hf f max"][f_idx]),
                  ]
        
        patient_idx = meta_info["patient_id"][f_idx]
        trial_idx = meta_info["trial"][f_idx]
        
        loc_lf_data = [values[0], meta_info["beta lfp strength 1"][f_idx], patient_idx, trial_idx]
        loc_hf_b_data = [values[1], meta_info["beta burst strength 1"][f_idx], patient_idx, trial_idx]
        loc_hf_nb_data = [values[2], meta_info["beta non burst strength 1"][f_idx], patient_idx, trial_idx]
        loc_dmi_b_data = [values[3], meta_info["beta burst strength 1"][f_idx], patient_idx, trial_idx]
        loc_dmi_nb_data = [values[4], meta_info["beta non burst strength 1"][f_idx], patient_idx, trial_idx]
        
        data_lf.append(loc_lf_data)
        data_hf_b.append(loc_hf_b_data)
        data_hf_nb.append(loc_hf_nb_data)
        data_pac_b.append(loc_dmi_b_data)
        data_pac_nb.append(loc_dmi_nb_data)
        
    return (data_lf, data_hf_b, data_hf_nb, data_pac_b, data_pac_nb)

import finn.statistical.glmm as glmm

def main(path, subpath, mode):
    data = get_values(path, subpath, mode)
    
    formula = "target_value ~ power + (1|patient_id) + (1|trial)"
    labels = ["target_value", "power", "patient_id", "trial"]
    factor_type = ["continuous", "continuous", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, power = contr.sum, patient_id = contr.sum, trial = contr.sum, fake = contr.sum)"
    data_type = "gaussian"
    
    for (tmp_idx, tmp) in enumerate(data):
        res = np.asarray(glmm.run(np.asarray(tmp, dtype = float), labels, factor_type, formula, contrasts, data_type))
        print(tmp_idx)
        for x in range(6):
            for y in range(2):
                print(res[x, y], end = "\t")
            print("")

main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/3/", "beta")


