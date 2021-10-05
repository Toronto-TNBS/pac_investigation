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
    (pre_labels, pre_data) = full_data.get_sheet_as_array("tremor")
    
    targets = ["strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    lf_tremor_idx = pre_labels.index("tremor lfp strength 1")
    hf_tremor_idx = pre_labels.index("tremor overall strength 1") 
    pac_burst_strength_idx = pre_labels.index("pac burst strength 2")
    spikes = pre_labels.index("spikes_per_second")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([spikes, lf_tremor_idx, hf_tremor_idx, pac_burst_strength_idx, patient_id_idx, trial_idx])
    
    idx_lists_burst = [idx_list_burst_0]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_burst[idx_list_idx]], [1]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            if (pre_labels[label_idx] == "tremor overall strength 1"):
                loc_labels.append("hf")
            elif(pre_labels[label_idx] == "tremor lfp strength 1"):
                loc_labels.append("lf")
            elif(pre_labels[label_idx] == "pac burst strength 2" or pre_labels[label_idx] == "pac non burst strength 2"):
                loc_labels.append("pac")
            else:
                loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        labels.append(loc_labels)

    data = np.asarray(data, dtype = np.float32)
    print(data.shape)
    data = orem.run(data, data[0, :, 0], 2.5, 5, 1)
    print(data.shape)
    
    formulas = ["target_value ~ lf + (1|patient_id) + (1|trial)", 
                "target_value ~ hf + (1|patient_id) + (1|trial)", 
                "target_value ~ pac + (1|patient_id) + (1|trial)", 
                "target_value ~ lf + hf + (1|patient_id) + (1|trial)", 
                "target_value ~ lf + hf + pac + (1|patient_id) + (1|trial)"]
    plot_idx = [1, 2, 3, None, None]
    
    #labels => ['target_value', 'lf', 'hf', 'patient id', 'trial', 'burst']
    factor_type = ["continuous", "continuous", "continuous", "continuous", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    print("Tremor")
    for (formula_idx, formula) in enumerate(formulas):
        for data_idx in range(len(data)):
            stats= glmm.run(data[data_idx], labels[data_idx], factor_type, formula, contrasts, data_type)
            print(np.asarray(stats))
            
            if (plot_idx[formula_idx] is not None):
                plt.figure()
                plt.scatter(data[data_idx][:, plot_idx[formula_idx]], data[data_idx][:, 0])
                plt.plot([np.min(data[data_idx][:, plot_idx[formula_idx]]), np.max(data[data_idx][:, plot_idx[formula_idx]])],
                         [stats[3][1] + stats[3][0] * np.min(data[data_idx][:, plot_idx[formula_idx]]), stats[3][1] +
                          stats[3][0] * np.max(data[data_idx][:, plot_idx[formula_idx]])])
                plt.title(formula)
            # (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
            # tmp = np.asarray(tmp); tmp[:5, :] = np.around(np.asarray(tmp[:5, :], dtype = np.float32), 4)
            #---------------------------------------------------- for x in range(6):
                #------------------------------------------------ for y in range(7):
                    #---------------------------------- print(tmp[x, y], end = "\t")
                #--------------------------------------------------------- print("")
            
            np.save("../../../../results/tremor/stats/77/stats_" + targets[data_idx] + ".npy", np.asarray(stats))
    plt.show(block = True)
    
    
main()
