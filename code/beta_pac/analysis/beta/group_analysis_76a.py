'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.glmm as glmm
import numpy as np
import os

import finn.cleansing.outlier_removal as orem

import matplotlib.pyplot as plt

def main():
    full_data = ods_reader.ods_data("../../../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = ["strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    lf_beta_idx = pre_labels.index("beta lfp strength 1")
    hf_beta_idx = pre_labels.index("beta overall strength 1") 
    pac_burst_strength_idx = pre_labels.index("pac burst strength 2")
    pac_non_burst_strength_idx = pre_labels.index("pac non burst strength 2")
    spikes = pre_labels.index("spikes_per_second")
    spikes_within = pre_labels.index("spikes_within_per_second")
    spikes_outside = pre_labels.index("spikes_outside_per_second")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([spikes_within, lf_beta_idx, hf_beta_idx, pac_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_non_burst_0 = np.asarray([spikes_outside, lf_beta_idx, hf_beta_idx, pac_non_burst_strength_idx, patient_id_idx, trial_idx])
    
    idx_lists_burst = [idx_list_burst_0]
    idx_lists_non_burst = [idx_list_non_burst_0]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            if (float(pre_data[row_idx, spikes]) < 20 or float(pre_data[row_idx, spikes]) > 60):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_non_burst[idx_list_idx]], [0]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            if (pre_labels[label_idx] == "beta overall strength 1"):
                loc_labels.append("hf")
            elif(pre_labels[label_idx] == "beta lfp strength 1"):
                loc_labels.append("lf")
            elif(pre_labels[label_idx] == "pac burst strength 2" or pre_labels[label_idx] == "pac non burst strength 2"):
                loc_labels.append("pac")
            else:
                loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        labels.append(loc_labels)

    data = np.asarray(data, dtype = np.float32)
    
    print(data.shape)
    data = orem.run(data, data[0, :, 0], 2, 5, 1)
    print(data.shape)
    
    formula = "target_value ~ hf + (1|patient_id) + (1|trial)"
    #labels => ['target_value', 'lf', 'hf', 'patient id', 'trial', 'burst']
    factor_type = ["continuous", "continuous", "continuous", "continuous", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    print("beta")
    for data_idx in range(len(data)):
        stats = glmm.run(data[data_idx], labels[data_idx], factor_type, formula, contrasts, data_type)
        print(np.asarray(stats))
            
        plt.figure()
        plt.scatter(data[data_idx][:, 2], data[data_idx][:, 0])
        plt.plot([np.min(data[data_idx][:, 2]), np.max(data[data_idx][:, 2])],
                 [stats[3][1] + stats[3][0] * np.min(data[data_idx][:, 2]), stats[3][1] +
                  stats[3][0] * np.max(data[data_idx][:, 2])])
        plt.title(formula)
        
        np.save("../../../../results/beta/stats/76/stats_" + targets[data_idx] + ".npy", np.asarray(stats))
    plt.show(block = True)
    
    
main()
