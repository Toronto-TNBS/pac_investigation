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
for (file_idx, _) in enumerate(data):
    for (trial_idx, _) in enumerate(data[file_idx][5]):
            
        if (data[file_idx][6][trial_idx] > 300):
            continue
        if (data[file_idx][7][trial_idx] > 100):
            continue
        
        stat_data.append([data[file_idx][5][trial_idx], data[file_idx][6][trial_idx], data[file_idx][7][trial_idx], file_idx, trial_idx])

stat_data = np.asarray(stat_data)

formula     = "power ~ burst_spike_freq + non_burst_spike_freq + (1|file_idx) + (1|trial_idx)"
labels      = ["power",         "burst_spike_freq", "non_burst_spike_freq", "file_idx",     "trial_idx"]
factor_type = ["continuous",    "continuous",       "continuous",           "categorical",  "categorical"] 
contrasts   = "list(power = contr.sum, spike_freq = contr.sum, burst = contr.sum, file_idx = contr.sum, trial_idx = contr.sum)"
data_type   = "gaussian"

stats = glmm.run(stat_data, labels, factor_type, formula, contrasts, data_type)
print(np.asarray(stats))

plt.figure()
plt.scatter(stat_data[:, 1], stat_data[:, 0])
plt.plot([np.min(stat_data[:, 1]), np.max(stat_data[:, 1])],
         [stats[3][2] + stats[3][0] * np.min(stat_data[:, 1]), stats[3][2] +
          stats[3][0] * np.max(stat_data[:, 1])])
plt.title(formula)


#plt.figure()
#plt.scatter(stat_data_burst[:, 0], stat_data_burst[:, 1], alpha = 0.1)

plt.show(block = True)
print("terminated")




