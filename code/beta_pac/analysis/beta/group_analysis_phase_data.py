'''
Created on Aug 31, 2021

@author: voodoocode
'''

import numpy as np
import finn.statistical.glmm as glmm
import methods.data_io.ods as ods_reader

full_data = ods_reader.ods_data("../../../../data/meta.ods")
(pre_labels, pre_data) = full_data.get_sheet_as_array("beta")
pre_data = np.asarray(pre_data)
path = "/mnt/data/Professional/UHN/pac_investigation/results/beta/pre/"

data = list()
for (file_idx, file) in enumerate(pre_data[:, 0]):
    if (int(pre_data[file_idx, pre_labels.index("valid_data")]) == 0):
        continue
    
    corr_beta_amp_0 = np.load(path + file + "/corr_beta_amp_0.npy")
    corr_beta_amp_1 = np.load(path + file + "/corr_beta_amp_1.npy")
    corr_beta_amp_2 = np.load(path + file + "/corr_beta_amp_2.npy")
    
    corr_hfo_amp_0 = np.load(path + file + "/corr_hfo_amp_0.npy")
    corr_hfo_amp_1 = np.load(path + file + "/corr_hfo_amp_1.npy")
    corr_hfo_amp_2 = np.load(path + file + "/corr_hfo_amp_2.npy")
    
    burst_type_0 = np.load(path + file + "/burst_type_0.npy")
    burst_type_1 = np.load(path + file + "/burst_type_1.npy")
    burst_type_2 = np.load(path + file + "/burst_type_2.npy")
    
    for data_pt_idx in range(len(corr_beta_amp_0)):
        data.append([corr_beta_amp_0[data_pt_idx], corr_hfo_amp_0[data_pt_idx], burst_type_0[data_pt_idx], 0,
                     data_pt_idx, pre_data[file_idx, pre_labels.index("patient_id")], pre_data[file_idx, pre_labels.index("hf auto")]])
    for data_pt_idx in range(len(corr_beta_amp_1)):
        data.append([corr_beta_amp_1[data_pt_idx], corr_hfo_amp_1[data_pt_idx], burst_type_1[data_pt_idx], 1,
                     data_pt_idx, pre_data[file_idx, pre_labels.index("patient_id")], pre_data[file_idx, pre_labels.index("hf auto")]])
    for data_pt_idx in range(len(corr_beta_amp_2)):
        data.append([corr_beta_amp_2[data_pt_idx], corr_hfo_amp_2[data_pt_idx], burst_type_2[data_pt_idx], 2,
                     data_pt_idx, pre_data[file_idx, pre_labels.index("patient_id")], pre_data[file_idx, pre_labels.index("hf auto")]])

data = np.asarray(data, dtype = float)

labels      = ['corr_beta_amp', 'corr_hfo_amp', 'burst', 'spike_cnt', 'trial', 'patient_id', 'beta']
factor_type = ["continuous", "continuous", "categorical", "categorical", "continuous", "categorical", "categorical"] 
contrasts   = "list(corr_beta_amp = contr.sum, corr_hfo_amp = contr.sum, burst = contr.sum, spike_cnt = contr.sum, trial = contr.sum, patient_id = contr.sum, beta = contr.sum)"
data_type   = "gaussian"

#beta & burst
formula     = "corr_beta_amp ~ spike_cnt + (1|patient_id) + (1|trial)"
loc_data    = np.copy(data)
loc_data    = loc_data[np.argwhere(loc_data[:, 2] == 1).squeeze(), :] # burst
loc_data    = loc_data[np.argwhere(loc_data[:, 6] == 1).squeeze(), :] # beta
res = glmm.run(loc_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(res))

#beta & non burst
formula     = "corr_beta_amp ~ spike_cnt + (1|patient_id) + (1|trial)"
loc_data    = np.copy(data)
loc_data    = loc_data[np.argwhere(loc_data[:, 2] == 0).squeeze(), :] # burst
loc_data    = loc_data[np.argwhere(loc_data[:, 6] == 1).squeeze(), :] # beta
res = glmm.run(loc_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(res))

#non beta & burst
formula     = "corr_beta_amp ~ spike_cnt + (1|patient_id) + (1|trial)"
loc_data    = np.copy(data)
loc_data    = loc_data[np.argwhere(loc_data[:, 2] == 1).squeeze(), :] # burst
loc_data    = loc_data[np.argwhere(loc_data[:, 6] == 0).squeeze(), :] # beta
res = glmm.run(loc_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(res))

#non beta & non burst
formula     = "corr_beta_amp ~ spike_cnt + (1|patient_id) + (1|trial)"
loc_data    = np.copy(data)
loc_data    = loc_data[np.argwhere(loc_data[:, 2] == 0).squeeze(), :] # burst
loc_data    = loc_data[np.argwhere(loc_data[:, 6] == 0).squeeze(), :] # beta
res = glmm.run(loc_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(res))



print("Terminated")














