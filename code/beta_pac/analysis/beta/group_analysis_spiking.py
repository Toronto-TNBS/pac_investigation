'''
Created on Oct 6, 2021

@author: voodoocode
'''

import numpy as np
import finn.statistical.glmm as glmm

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import methods.data_io.ods as ods_reader

data = np.load("amp_spike_data.npy", allow_pickle = True)
meta_data = ods_reader.ods_data("../../../../data/meta.ods")
(pre_labels, pre_data) = meta_data.get_sheet_as_array("beta")
meta_hf_auto_idx = pre_labels.index("hf auto")
meta_file_idx = pre_labels.index("file")

stat_data = list()
stat_data_burst = list()
stat_data_non_burst = list()
for (file_idx, _) in enumerate(data):
    #Add burst data
    for (trial_idx, _) in enumerate(data[file_idx][1]):
        
        if (data[file_idx][2][trial_idx] > 300):
            continue
        
        loc_burst_stat_data = [data[file_idx][1][trial_idx], data[file_idx][2][trial_idx], 1, file_idx, trial_idx, int(float(pre_data[pre_data[:, 0].tolist().index(data[file_idx][5]), meta_hf_auto_idx]))]
        stat_data.append(loc_burst_stat_data)
        stat_data_burst.append(loc_burst_stat_data)
        
    for (trial_idx, _) in enumerate(data[file_idx][3]):
        
        if (data[file_idx][4][trial_idx] > 300):
            continue
        
        loc_burst_stat_data = [data[file_idx][3][trial_idx], data[file_idx][4][trial_idx], 0, file_idx, trial_idx, int(float(pre_data[pre_data[:, 0].tolist().index(data[file_idx][5]), meta_hf_auto_idx]))]
        stat_data.append(loc_burst_stat_data)
        stat_data_non_burst.append(loc_burst_stat_data)
stat_data = np.asarray(stat_data)
stat_data_burst = np.asarray(stat_data_burst)
stat_data_non_burst = np.asarray(stat_data_non_burst)

formula     = "power ~ spike_freq + burst + spike_freq:burst + (1|file_idx) + (1|trial_idx)"
#formula     = "spike_freq ~ power + burst + spike_freq:burst + (1|file_idx) + (1|trial_idx)"
labels      = ["power",         "spike_freq",   "burst",        "file_idx",     "trial_idx",    "beta"]
factor_type = ["continuous",    "continuous",   "categorical",  "categorical",  "categorical",  "categorical"] 
contrasts   = "list(power = contr.sum, spike_freq = contr.sum, burst = contr.sum, file_idx = contr.sum, trial_idx = contr.sum)"
data_type   = "gaussian"

stats = glmm.run(stat_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(stats))

formula     = "power ~ spike_freq + (1|file_idx) + (1|trial_idx)"
#formula     = "spike_freq ~ power + (1|file_idx) + (1|trial_idx)"
stats = glmm.run(stat_data_burst, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(stats))

stats = glmm.run(stat_data_non_burst, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(stats))

plt.figure()
plt.scatter(stat_data_burst[:, 0], stat_data_burst[:, 1], alpha = 0.1)

plt.figure()
plt.scatter(stat_data_non_burst[:, 0], stat_data_non_burst[:, 1], alpha = 0.1)

plt.show(block = True)
print("termianted")




