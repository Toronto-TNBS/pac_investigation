'''
Created on May 21, 2021

@author: voodoocode
'''

import methods.data_io.ods as ods_reader
import finn.statistical.generalized_linear_models as glmm
import numpy as np

def main():
    full_data = ods_reader.ods_data("../../../../data/meta_simp.ods")
    (pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
    
    targets = [
               #================================================================
               # "tremor_lf", "tremor_hf", "tremor_hf_b", "tremor_hf_n_b",
               # "tremor_pac_b", "tremor_pac_nb", "tremor_pac_spec_b", "tremor_pac_spec_nb",
               # "rigidity_lf", "rigidity_hf", "rigidity_hf_b", "rigidity_hf_n_b",
               # "rigidity_pac_b", "rigidity_pac_nb", "rigidity_pac_spec_b", "rigidity_pac_spec_nb",
               #================================================================
               "total_motor_updrs_lf", "total_motor_updrs_hf", "total_motor_updrs_hf_b", "total_motor_updrs_hf_nb",
               "total_motor_updrs_pac_b", "total_motor_updrs_pac_nb", "total_motor_updrs_pac_spec_b", "total_motor_updrs_pac_spec_nb"]
    patient_id_idx = pre_labels.index("patient_id")
    trial_idx = pre_labels.index("trial")
    valid_idx = pre_labels.index("valid_data")
    target_idx = pre_labels.index("target")
    lf_beta_idx = pre_labels.index("beta lfp strength 1")
    hf_beta_idx = pre_labels.index("beta overall strength 1")
    hf_burst_beta_idx = pre_labels.index("beta burst strength 1")
    hf_non_burst_beta_idx = pre_labels.index("beta non burst strength 1")
    pac_burst_strength_idx = pre_labels.index("pac burst strength 3")
    pac_non_burst_strength_idx = pre_labels.index("pac non burst strength 3")
    pac_burst_specificity_idx = pre_labels.index("pac burst specificity 3")
    pac_non_burst_specificity_idx = pre_labels.index("pac non burst specificity 3")
    pac_burst_specific_strenght_idx = pre_labels.index("pac burst specific strength 3")
    pac_non_burst_specific_strenght_idx = pre_labels.index("pac non burst specific strength 3")
    tremor_idx = pre_labels.index("tremor")
    rigidity_idx = pre_labels.index("rigidity")
    total_motor_updrs_idx = pre_labels.index("total_motor_updrs")
    pre_labels = [pre_label.replace(" auto","") if (type(pre_label) == str) else pre_label for pre_label in pre_labels]

#===============================================================================
#     idx_list_burst_00 = np.asarray([tremor_idx, patient_id_idx, trial_idx, lf_beta_idx])
#     idx_list_burst_01 = np.asarray([tremor_idx, patient_id_idx, trial_idx, hf_beta_idx])
#     idx_list_burst_02 = np.asarray([tremor_idx, patient_id_idx, trial_idx, hf_burst_beta_idx])
#     idx_list_burst_03 = np.asarray([tremor_idx, patient_id_idx, trial_idx, hf_non_burst_beta_idx])
#     idx_list_burst_04 = np.asarray([tremor_idx, patient_id_idx, trial_idx, pac_burst_strength_idx])
#     idx_list_burst_05 = np.asarray([tremor_idx, patient_id_idx, trial_idx, pac_non_burst_strength_idx])
#     idx_list_burst_06 = np.asarray([tremor_idx, patient_id_idx, trial_idx, pac_burst_specific_strenght_idx])
#     idx_list_burst_07 = np.asarray([tremor_idx, patient_id_idx, trial_idx, pac_non_burst_specific_strenght_idx])
# 
#     idx_list_burst_10 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, lf_beta_idx])
#     idx_list_burst_11 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, hf_beta_idx])
#     idx_list_burst_12 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, hf_burst_beta_idx])
#     idx_list_burst_13 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, hf_non_burst_beta_idx])
#     idx_list_burst_14 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, pac_burst_strength_idx])
#     idx_list_burst_15 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, pac_non_burst_strength_idx])
#     idx_list_burst_16 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, pac_burst_specific_strenght_idx])
#     idx_list_burst_17 = np.asarray([rigidity_idx, patient_id_idx, trial_idx, pac_non_burst_specific_strenght_idx])
#===============================================================================

    idx_list_burst_20 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, lf_beta_idx])
    idx_list_burst_21 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, hf_beta_idx])
    idx_list_burst_22 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, hf_burst_beta_idx])
    idx_list_burst_23 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, hf_non_burst_beta_idx])
    idx_list_burst_24 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, pac_burst_strength_idx])
    idx_list_burst_25 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, pac_non_burst_strength_idx])
    idx_list_burst_26 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, pac_burst_specific_strenght_idx])
    idx_list_burst_27 = np.asarray([total_motor_updrs_idx, patient_id_idx, trial_idx, pac_non_burst_specific_strenght_idx])
    
    idx_lists = [
                 #==============================================================
                 # idx_list_burst_00, idx_list_burst_01, idx_list_burst_02, idx_list_burst_03, 
                 # idx_list_burst_04, idx_list_burst_05, idx_list_burst_06, idx_list_burst_07,
                 # idx_list_burst_10, idx_list_burst_11, idx_list_burst_12, idx_list_burst_13, 
                 # idx_list_burst_14, idx_list_burst_15, idx_list_burst_16, idx_list_burst_17,
                 #==============================================================
                 idx_list_burst_20, idx_list_burst_21, idx_list_burst_22, idx_list_burst_23, 
                 idx_list_burst_24, idx_list_burst_25, idx_list_burst_26, idx_list_burst_27]
    
    data = list()
    for idx_list_idx in range(len(idx_lists)):
        data.append(list())
        for row_idx in range(len(pre_data)):
            if (int(pre_data[row_idx, valid_idx]) == 0):
                continue
            
            if ("N/A" in pre_data[row_idx, idx_lists[idx_list_idx]]):
                continue
            
            if (pre_data[row_idx, target_idx] == "GPi"):
                continue
            
            loc_data = np.concatenate((pre_data[row_idx, idx_lists[idx_list_idx]], [0]))
            data[-1].append(loc_data)

    data = np.asarray(data, dtype = np.float32)
    
    formula = "exp_value ~ target_value + (1|patient_id) + (1|trial)"
    formula = "target_value ~ exp_value"
    labels = ['target_value', 'patient_id', 'trial', 'exp_value']
    factor_type = ["continuous", "categorical", "categorical", "continuous"] 
    contrasts = "list(target_value = contr.sum, patient_id = contr.sum, trial = contr.sum, exp_value = contr.sum)"
    data_type = "gaussian"
    
    for data_idx in range(len(data)):
        tmp = glmm.run(data[data_idx], labels, factor_type, formula, contrasts, data_type)
        (chi_sq_scores, df, p_values, coefficients, std_error, factor_names) = tmp
        
        print(targets[data_idx], float(np.asarray(tmp)[2, 0]) < 0.05, np.asarray(tmp)[2, 0], np.asarray(tmp)[-1, 0])
        
        np.save("../../../../results/beta/stats/4/stats_" + targets[data_idx] + ".npy", np.asarray(tmp))
    
    
main()
