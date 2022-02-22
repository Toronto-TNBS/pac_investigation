'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.glmm as glmm
import numpy as np
import os

import finn.cleansing.outlier_removal as orem

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

def main():
    full_data = ods_reader.ods_data("../../../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = ["strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    lf_beta_idx = pre_labels.index("beta lfp strength 1")
    hf_beta_idx = pre_labels.index("beta overall strength 1")
    hf_beta_burst_idx = pre_labels.index("beta burst strength 1")
    hf_beta_non_burst_idx = pre_labels.index("beta non burst strength 1")
    spike_freq_idx = pre_labels.index("spikes_per_second")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]
 
    idx_list = np.asarray([spike_freq_idx, lf_beta_idx, hf_beta_idx, hf_beta_burst_idx, hf_beta_non_burst_idx, patient_id_idx, trial_idx])
    
    data = list()
    for row_idx in range(len(pre_data)):
        
        if (int(pre_data[row_idx, valid_idx]) == 0):
            continue
        
        if (float(pre_data[row_idx, spike_freq_idx]) < 10 or float(pre_data[row_idx, spike_freq_idx]) > 90):
            continue
        
        data.append(pre_data[row_idx, idx_list])

    data = np.asarray(data, dtype = np.float32)
    
    formulas = ["spike_freq ~ hf_pow + (1|patient_id) + (1|trial)", "spike_freq ~ hf_burst_pow + (1|patient_id) + (1|trial)", "spike_freq ~ hf_non_burst_pow + (1|patient_id) + (1|trial)",
                "lf_pow ~ hf_pow + (1|patient_id) + (1|trial)", "lf_pow ~ hf_burst_pow + (1|patient_id) + (1|trial)", "lf_pow ~ hf_non_burst_pow + (1|patient_id) + (1|trial)",]
    labels = ["spike_freq", "lf_pow", "hf_pow", "hf_burst_pow", "hf_non_burst_pow", "patient_id", "trial"]
    factor_type = ["continuous", "continuous", "continuous", "continuous", "continuous", "categorical", "categorical"] 
    contrasts = "list(spike_freq = contr.sum, lf_pow = contr.sum, hf_pow = contr.sum, hf_burst_pow = contr.sum, hf_non_burst_pow = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    for (formula_idx, formula) in enumerate(formulas):
        loc_data = np.copy(data)
        print(loc_data.shape)
        
        if (formula_idx < 3):
            loc_data = loc_data[np.argwhere(loc_data[:, formula_idx + 2] > -1).squeeze(), :]
            loc_data = loc_data[np.argwhere(loc_data[:, formula_idx + 2] < 1).squeeze(), :]
        
        if ("hf_burst_pow" in formula):
            loc_data = loc_data[np.argwhere(loc_data[:, 3] != -1).squeeze(), :]
        stats = np.asarray(glmm.run(loc_data, labels, factor_type, formula, contrasts, data_type)[:-1], dtype = np.float32)
        print(float(stats[2, 0])*len(formulas), "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])), float(stats[3, 0]) + float(stats[3, 1]), float(stats[3, 1]))


        plt.figure()
        plt.title(formula)
        y_vals = loc_data[:, int(formula_idx/3)]; x_vals = loc_data[:, int(formula_idx%3+2)] 
        plt.scatter(x_vals, y_vals)
        plt.plot([np.min(x_vals), np.max(x_vals)], [stats[3,1], stats[3,1] + stats[3,0]])
        plt.title(formula)
    plt.show(block = True)
    
    
main()
