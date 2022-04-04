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

import lmfit

hfo_offset = 3
burst_spike_offset = 12
non_burst_spike_offset = 21

idx_list = [hfo_offset, burst_spike_offset, non_burst_spike_offset]

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/projects/old/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    data_list = list()
    sin_fit_list = list()
    patients = list()
    trials = list()
    pac_scores = list()
    
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (type == "hf beta" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (type == "hf non beta" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
               
        data_list.append([data[idx + 1] for idx in idx_list])
        pac_scores.append([data[idx + 2] for idx in idx_list])
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        sin_fit_list.append([data[hfo_offset], data[burst_spike_offset], data[non_burst_spike_offset]])

    pac_scores = np.asarray(pac_scores)
       
    return (np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials), pac_scores)

def main(path, subpath, mode, type, axes, ideal_ref_slope = None):
    (data, fit, patient_ids, trial_ids, pac_scores) = get_values(path, subpath, mode, type)
    
    cond_data = list()
    for cond_idx in range(data.shape[1]):
        
        loc_patients = np.copy(patient_ids)
        loc_trials = np.copy(trial_ids)
    
        loc_data = data[:, cond_idx, :]; loc_fit = fit[:, cond_idx, :]; loc_pac_scores = np.copy(pac_scores[:, cond_idx])
        
        loc_data_m = np.mean(loc_data, axis = 0)
        loc_data_v = np.sqrt(np.var(loc_data, axis = 0))
        
        def __sine(x, phase, amp):
            """
            Internal method. Used in run_dmi to estimate the direct modulation index. The amount of PAC is quantified via a sine fit. This sine is defined by the following paramters:
            
            :param x: Samples
            :param phase: Phase shift of the sine.
            :param amp: Amplitude of the sine.
            
            :return: Returns the fitted sine at the locations indicated by x.
            """
            freq = 1
            fs = 1
            return amp * (np.sin(2 * np.pi * freq * (x - ((phase + 180)/360)) / fs))
        amplitude_signal = np.mean(loc_data, axis = 0)
        params = lmfit.Parameters()
        params.add("phase", value = 0, min = -180, max = 180, vary = True)
        params.add("amp", value = 0.5, min = .9, max = 1.1, vary = True)
        model = lmfit.Model(__sine, nan_policy = "omit")
        result = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
                 
        axes[cond_idx].plot(result.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
        axes[cond_idx].plot(loc_data_m, color = "black", label = "avg. burst", zorder = 3)
        axes[cond_idx].fill_between(np.arange(0, loc_data.shape[1]),
                             loc_data_m + loc_data_v, 
                             loc_data_m - loc_data_v, color = "green", alpha = 0.25, zorder = 1)
        axes[cond_idx].set_ylim((-2.5, 2.5))
        
        loc_sq_errors = np.sum(np.power((loc_data - result.best_fit), 2), axis = 1)
        
        cond_data.append(np.asarray([loc_pac_scores, loc_sq_errors, loc_patients, loc_trials]))
           
    return np.asarray(cond_data)

(fig, axes) = plt.subplots(1, 3)
cond_data = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf beta", axes)

data1 = np.asarray(cond_data[0, :, :], dtype = np.float32).transpose();
data2 = np.asarray(cond_data[1, :, :], dtype = np.float32).transpose();
data3 = np.asarray(cond_data[2, :, :], dtype = np.float32).transpose();

stat_data_21 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data1, np.ones((data1.shape[0], 1))), axis = 1)), axis = 0)
stat_data_23 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data3, np.ones((data3.shape[0], 1))), axis = 1)), axis = 0)

stat_data = [stat_data_21, stat_data_23]

for loc_stat_data in stat_data:
    stats = glmm.run(loc_stat_data, ["pac_score", "shape_score", "patient", "trial", "type"], ["continuous", "continuous", "categorical", "categorical", "categorical"],
                     "pac_score ~ type + (1|patient) + (1|trial)", "list(pac_score = contr.sum, shape_score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                     "gaussian")
    stats = np.asarray(stats)
    #print(stats)
    print(float(stats[2, 0])*1, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])), float(stats[3, 0]) + float(stats[3, 1]), float(stats[3, 1]))

plt.tight_layout()
plt.show(block = True)



