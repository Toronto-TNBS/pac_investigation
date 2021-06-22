'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.generalized_linear_models as glmm
import numpy as np

def main3():
    full_data = ods_reader.ods_data("../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("tremor")
    
    targets = ["lf tremor", "hf tremor", "hf burst tremor", "hf non burst tremor"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    lf_tremor_idx = pre_labels.index("lf manual")
    hf_tremor_idx = pre_labels.index("hf manual")
    valid_idx = pre_labels.index("valid_data")
    tremor_lfp_strength_idx = pre_labels.index("tremor lfp strength 1")
    tremor_spike_strength_idx = pre_labels.index("tremor overall strength 1")
    tremor_burst_strength_idx = pre_labels.index("tremor burst strength 1")
    tremor_non_burst_strength_idx = pre_labels.index("tremor non burst strength 1")
    pre_labels = [pre_label.replace(" manual","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([tremor_lfp_strength_idx, patient_id_idx, trial_idx, lf_tremor_idx, hf_tremor_idx])
    idx_list_burst_1 = np.asarray([tremor_spike_strength_idx, patient_id_idx, trial_idx, lf_tremor_idx, hf_tremor_idx])
    idx_list_burst_2 = np.asarray([tremor_burst_strength_idx, patient_id_idx, trial_idx, lf_tremor_idx, hf_tremor_idx])
    idx_list_burst_3 = np.asarray([tremor_non_burst_strength_idx, patient_id_idx, trial_idx, lf_tremor_idx, hf_tremor_idx])
    
    idx_lists_burst = [idx_list_burst_0, idx_list_burst_1, idx_list_burst_2, idx_list_burst_3]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            if (idx_list_idx == 0 and int(pre_data[row_idx, lf_tremor_idx]) == -1):# Discard samples which could not be manually classified
                continue
            if (idx_list_idx > 0 and int(pre_data[row_idx, hf_tremor_idx]) == -1): # Discard samples which could not be manually classified
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_burst[idx_list_idx]], [0]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        labels.append(loc_labels)

    for data_idx in range(len(data)):
        data[data_idx] = np.asarray(data[data_idx], np.float32)
    
    formula = ["target_value ~ lf + (1|patient_id) + (1|trial)", 
               "target_value ~ hf + (1|patient_id) + (1|trial)", 
               "target_value ~ hf + (1|patient_id) + (1|trial)", 
               "target_value ~ hf + (1|patient_id) + (1|trial)"]
    #labels => ['target_value', 'patient id', 'trial', 'lf_3', 'hf_3', 'burst']
    factor_type = ["continuous", "categorical", "categorical", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx], labels[data_idx], factor_type, formula[data_idx], contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print(targets[data_idx], np.asarray(tmp))
        
        np.save("../../results/tremor/stats/1/stats_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    
main3()
