'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.glmm as glmm
import numpy as np
import os

#Give continuous beta metric a try!

def main():
    full_data = ods_reader.ods_data("../../../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = ["strength", "specificity", "specific strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    lf_beta_idx = pre_labels.index("lf auto")
    lf_beta_idx = pre_labels.index("beta lfp strength 1")
    hf_beta_idx = pre_labels.index("hf auto")
    hf_beta_idx = pre_labels.index("beta overall strength 1")
    pac_burst_strength_idx = pre_labels.index("pac burst strength 2")
    pac_non_burst_strength_idx = pre_labels.index("pac non burst strength 2")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([pac_burst_strength_idx, lf_beta_idx, hf_beta_idx, patient_id_idx, trial_idx])
    idx_list_non_burst_0 = np.asarray([pac_non_burst_strength_idx, lf_beta_idx, hf_beta_idx, patient_id_idx, trial_idx])
    
    idx_lists_burst = [idx_list_burst_0]
    idx_lists_non_burst = [idx_list_non_burst_0]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_burst[idx_list_idx]], [1]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_non_burst[idx_list_idx]], [0]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            if (pre_labels[label_idx] == "beta overall strength 1"):
                loc_labels.append("hf")
            elif(pre_labels[label_idx] == "beta lfp strength 1"):
                loc_labels.append("lf")
            else:
                loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        labels.append(loc_labels)

    data = np.asarray(data, dtype = np.float32)
    #print(data[0, :, 2].shape)
    #data[0, :, 2] = np.concatenate((np.zeros((133)), np.ones((133))))# np.random.permutation(np.random.random_sample(data[0, :, 2].shape[0]))
    #np.save("tmp2.npy", data)
    
    
    formula = "target_value ~ burst + lf + hf + burst:lf + burst:hf + lf:hf + (1|patient_id) + (1|trial)"
    #formula = "target_value ~ burst + hf + burst:hf + (1|patient_id) + (1|trial)"
    #formula = "target_value ~ burst + lf + burst:lf + (1|patient_id) + (1|trial)"
    #formula = "target_value ~ lf + (1|patient_id) + (1|trial)"
    #labels => ['target_value', 'lf', 'hf', 'patient id', 'trial', 'burst']
    factor_type = ["continuous", "continuous", "continuous", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        tmp = np.asarray(tmp); tmp[:5, :] = np.around(np.asarray(tmp[:5, :], dtype = np.float32), 4) 
        for x in range(6):
            for y in range(7):
                print(tmp[x, y], end = "\t")
            print("")
        
        np.save("../../../../results/beta/stats/25/stats_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    
main()
