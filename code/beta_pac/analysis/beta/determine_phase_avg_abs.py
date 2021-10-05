'''
Created on Jun 10, 2021

@author: voodoocode
'''


import pickle
import numpy as np
import methods.data_io.ods

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.stats
import finn.cleansing.outlier_removal as orem

import finn.statistical.glmm as glmm

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    
    patients = list()
    trials = list()
    
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (type == "hf beta" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (type == "hf non beta" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        phase_shifts.append(np.argmin(data[1]))
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append(data[2])
        sin_fit_list.append(data[1])

       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials))

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, type, axes):
    (values, data, fit, patients, trials) = get_values(path, subpath, mode, type)
    patients = np.copy(patients)
    trials = np.copy(trials)
    
    data = out_rem.run(data, np.argmax(fit, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    patients = out_rem.run(patients, np.argmax(fit, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    trials = out_rem.run(trials, np.argmax(fit, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    fit = out_rem.run(fit, np.argmax(fit, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    
    best_fit = np.mean(fit, axis = 0)
    
    sq_error = np.sum(np.power((data - best_fit), 2), axis = 1)
    axes.plot(np.arange(-180, 181, 1), best_fit, "--", color = "red", zorder = 3, alpha = 0.5, label = "lfp")
    axes.plot(np.arange(-180, 181, 1), np.mean(data, axis = 0), color = "black", label = "avg. burst", zorder = 2)
    for x in range(data.shape[0]):
        axes.plot(np.arange(-180, 181, 1), data[x, :], color = "green", alpha = 0.25, zorder = 1)
    
    axes.set_ylim((-2.5, 2.5))
        
    return (np.concatenate((np.expand_dims(sq_error, axis = 1), np.expand_dims(patients, axis = 1), np.expand_dims(trials, axis = 1)), axis = 1))

(fig, axes) = plt.subplots(2, 1)
data1 = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/absolute_pac/", "beta", "hf beta", axes[0])
data2 = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/absolute_pac/", "beta", "hf non beta", axes[1])

data1 = np.asarray(data1, dtype = np.float32); data2 = np.asarray(data2, dtype = np.float32);

stat_data_12 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data2, np.ones((data2.shape[0], 1))), axis = 1)), axis = 0)

stats = glmm.run(stat_data_12, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
print(np.asarray(stats))

axes[0].set_title("Highly beta")
axes[1].set_title("Little beta")
plt.tight_layout()
plt.show(block = True)



