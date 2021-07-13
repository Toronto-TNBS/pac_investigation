'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.glmm as glmm
import numpy as np

def main1():
    full_data = ods_reader.ods_data("../../../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = ["strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    beta_burst_strength_idx = pre_labels.index("pac burst specific strength norm 3")
    beta_non_burst_strength_idx = pre_labels.index("pac non burst specific strength norm 3")
    beta_random_burst_strength_idx = pre_labels.index("pac random burst specific strength norm 3")
    beta_random_non_burst_strength_idx = pre_labels.index("pac random non burst specific strength norm 3")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([beta_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_non_burst_0 = np.asarray([beta_non_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_random_burst_0 = np.asarray([beta_random_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_random_non_burst_0 = np.asarray([beta_random_non_burst_strength_idx, patient_id_idx, trial_idx])
    
    idx_lists_burst = [idx_list_burst_0]
    idx_lists_non_burst = [idx_list_non_burst_0]
    idx_lists_random_burst = [idx_list_random_burst_0]
    idx_lists_random_non_burst = [idx_list_random_non_burst_0]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_burst[idx_list_idx]], [1], [1]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_non_burst[idx_list_idx]], [0], [1]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_random_burst[idx_list_idx]], [1], [0]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_random_non_burst[idx_list_idx]], [0], [0]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        loc_labels.append("structured")
        labels.append(loc_labels)

    data = np.asarray(data, dtype = np.float32)
    
    formula = "target_value ~ burst + (1|patient_id) + (1|trial)"
    formula = "target_value ~ burst + structured + burst:structured + (1|patient_id) + (1|trial)"
    #formula = "target_value ~ burst + structured + (1|patient_id) + (1|trial)"
    #labels => ['target_value', 'patient id', 'trial', 'burst']
    factor_type = ["continuous", "categorical", "categorical", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print(targets[data_idx], np.asarray(tmp))
        
        np.save("../../../../results/beta/stats/17/stats_" + targets[data_idx] + ".npy", np.asarray(tmp))


def main2():
    full_data = ods_reader.ods_data("../../../../data/meta.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = ["strength"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    beta_burst_strength_idx = pre_labels.index("pac burst specific strength norm 3")
    beta_non_burst_strength_idx = pre_labels.index("pac non burst specific strength norm 3")
    beta_random_burst_strength_idx = pre_labels.index("pac random burst specific strength norm 3")
    beta_random_non_burst_strength_idx = pre_labels.index("pac random non burst specific strength norm 3")
    valid_idx = pre_labels.index("valid_data")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

    idx_list_burst_0 = np.asarray([beta_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_non_burst_0 = np.asarray([beta_non_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_random_burst_0 = np.asarray([beta_random_burst_strength_idx, patient_id_idx, trial_idx])
    idx_list_random_non_burst_0 = np.asarray([beta_random_non_burst_strength_idx, patient_id_idx, trial_idx])
    
    idx_lists_burst = [idx_list_burst_0]
    idx_lists_non_burst = [idx_list_non_burst_0]
    idx_lists_random_burst = [idx_list_random_burst_0]
    idx_lists_random_non_burst = [idx_list_random_non_burst_0]
    
    data = list()
    labels = list()
    for idx_list_idx in range(len(idx_lists_burst)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_burst[idx_list_idx]], [1], [1]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_non_burst[idx_list_idx]], [0], [1]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_random_burst[idx_list_idx]], [1], [0]))
            data[-1].append(loc_data)
            loc_data = np.concatenate((pre_data[row_idx, idx_lists_random_non_burst[idx_list_idx]], [0], [0]))
            data[-1].append(loc_data)
        loc_labels = list(); loc_labels.append("target_value") 
        for label_idx in idx_lists_burst[idx_list_idx][1:]:
            loc_labels.append(pre_labels[label_idx])
        loc_labels.append("burst")
        loc_labels.append("structured")
        labels.append(loc_labels)

    data = np.asarray(data, dtype = np.float32)
    
    #formula = "target_value ~ burst + structured + (1|patient_id) + (1|trial)"
    #labels => ['target_value', 'patient id', 'trial', 'burst']
    factor_type = ["continuous", "categorical", "categorical", "categorical", "categorical", "categorical"] 
    contrasts = "list(target_value = contr.sum, lf = contr.sum, hf = contr.sum, burst = contr.sum, patient_id = contr.sum, trial = contr.sum)"
    data_type = "gaussian"
    
    formula = "target_value ~ structured + (1|patient_id) + (1|trial)"
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx, np.argwhere(data[data_idx, :, -2] == 1).squeeze(), :], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print("burst", targets[data_idx], np.asarray(tmp))
        
        np.save("../../../../results/beta/stats/17/stats_b_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx, np.argwhere(data[data_idx, :, -2] == 0).squeeze(), :], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print("non burst", targets[data_idx], np.asarray(tmp))
        
        np.save("../../../../results/beta/stats/17/stats_nb_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    formula = "target_value ~ burst + (1|patient_id) + (1|trial)"
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx, np.argwhere(data[data_idx, :, -1] == 1).squeeze(), :], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print("structured", targets[data_idx], np.asarray(tmp))
        
        np.save("../../../../results/beta/stats/17/stats_s_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx, np.argwhere(data[data_idx, :, -1] == 0).squeeze(), :], labels[data_idx], factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print("non structured", targets[data_idx], np.asarray(tmp))
        
        np.save("non structured", "../../../../results/beta/stats/17/stats_ns_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
main1()
main2()
