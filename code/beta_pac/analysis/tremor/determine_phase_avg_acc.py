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

import os

import lmfit

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list()
    
    patients = list()
    trials = list()
    
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (type == "hf tremor" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (type == "hf non tremor" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
        
        if (os.path.exists(path + mode + subpath + f_name + ".pkl") == False):
            continue
        
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[4])), np.argmin(np.abs(data[7])))
        
#        loc_data = (np.argmax(data[0]), np.argmax(data[3]), np.argmax(data[6]))
        phase_shifts.append(loc_data)
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append([data[1], data[4], data[7]])
        sin_fit_list.append([data[0], data[3], data[6]])
        fit_list0.append(data[1]); fit_list1.append(data[4]); fit_list2.append(data[7])

    fit_list0 = np.asarray(fit_list0); fit_list1 = np.asarray(fit_list1); fit_list2 = np.asarray(fit_list2)
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials))

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, type, axes):
    (values, data, fit, patients, trials) = get_values(path, subpath, mode, type)
    patients1 = np.copy(patients); patients2 = np.copy(patients)
    trials1 = np.copy(trials); trials2 = np.copy(trials)
    
    loc_data1 = data[:, 1, :]; loc_fit1 = fit[:, 1, :]
    loc_data2 = data[:, 2, :]; loc_fit2 = fit[:, 2, :]
    
    pre = (loc_data1.shape[0], loc_data2.shape[0])
    
    loc_data1 = out_rem.run(loc_data1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    patients1 = out_rem.run(patients1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    trials1 = out_rem.run(trials1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    loc_data2 = out_rem.run(loc_data2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    patients2 = out_rem.run(patients2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    trials2 = out_rem.run(trials2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    
    print("%2.2f | %2.2f" % (np.float32(loc_data1.shape[0]/pre[0]), np.float32(loc_data2.shape[0]/pre[1]),))
    
    loc_data1_m = np.mean(loc_data1, axis = 0)
    loc_data2_m = np.mean(loc_data2, axis = 0)
    min_val = np.min([loc_data1_m, loc_data2_m]); loc_data1_m -= min_val; loc_data2_m -= min_val
    max_val = np.max([loc_data1_m, loc_data2_m]); loc_data1_m /= max_val; loc_data2_m /= max_val
    loc_data1_m -= 0.5; loc_data1_m *= 2; loc_data2_m -= 0.5; loc_data2_m *= 2
        
    tmp = 361
    ideal_slope = np.sin(2 * np.pi * 1 * np.arange(0, tmp)/tmp)
    
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
        return amp * (np.sin(2 * np.pi * freq * (x - phase) / fs))
    
    ref_data_1 = np.average(loc_data1, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    ideal_slope_1 = model.fit(ref_data_1, x = np.arange(0, 1, 1/len(ref_data_1)),
                       params = params, max_nfev = 300).best_fit
                    
    ref_data_2 = np.average(loc_data2, axis = 0)   
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    ideal_slope_2 = model.fit(ref_data_2, x = np.arange(0, 1, 1/len(ref_data_2)),
                       params = params, max_nfev = 300).best_fit
    
    sq_error_1 = np.sum(np.power((loc_data1 - ideal_slope_1), 2), axis = 1)
    loc_data1 = orem.run(np.copy(loc_data1), sq_error_1, 2, 5, 0)
    patients1 = orem.run(np.copy(patients1), sq_error_1, 2, 5, 0)
    trials1 = orem.run(np.copy(trials1), sq_error_1, 2, 5, 0)
    sq_error_2 = np.sum(np.power((loc_data2 - ideal_slope_2), 2), axis = 1)
    loc_data2 = orem.run(np.copy(loc_data2), sq_error_2, 2, 5, 0)
    patients2 = orem.run(np.copy(patients2), sq_error_2, 2, 5, 0)
    trials2 = orem.run(np.copy(trials2), sq_error_2, 2, 5, 0)
    
    sq_error_1 = np.sum(np.power((loc_data1 - ideal_slope_1), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_data2 - ideal_slope_2), 2), axis = 1)
    print(loc_data1.shape)
    print(loc_data2.shape)
     
    axes[0].plot(ideal_slope_1, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(ideal_slope_2, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[0].plot(loc_data1_m, color = "black", label = "avg. burst", zorder = 3)
    for x in range(loc_data1.shape[0]):
        axes[0].plot(loc_data1[x, :], color = "green", alpha = 0.25, zorder = 1)
    axes[1].plot(loc_data2_m, color = "black", label = "avg. non burst", zorder = 3)
    for x in range(loc_data2.shape[0]):
        axes[1].plot(loc_data2[x, :], color = "blue", alpha = 0.25, zorder = 1)
    
    axes[0].set_ylim((-2.5, 2.5))    
    axes[1].set_ylim((-2.5, 2.5))
    #axes[0].legend()
    #axes[1].legend()
    
    print("all", np.mean(values[:, 0]), np.sqrt(np.var(values[:, 0])))
    print("burst", np.mean(values[:, 1]), np.sqrt(np.var(values[:, 1])))
    print("non burst", np.mean(values[:, 2]), np.sqrt(np.var(values[:, 2])))
    
    return (np.concatenate((np.expand_dims(sq_error_1, axis = 1), np.expand_dims(patients1, axis = 1), np.expand_dims(trials1, axis = 1)), axis = 1),
            np.concatenate((np.expand_dims(sq_error_2, axis = 1), np.expand_dims(patients2, axis = 1), np.expand_dims(trials2, axis = 1)), axis = 1))

(fig, axes) = plt.subplots(2, 2)
(data1, data2) = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/4/", "tremor", "hf tremor", axes[0, :])
(data3, data4) = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/4/", "tremor", "hf non tremor", axes[1, :])

data1 = np.asarray(data1, dtype = np.float32); data2 = np.asarray(data2, dtype = np.float32);
data3 = np.asarray(data3, dtype = np.float32); data4 = np.asarray(data4, dtype = np.float32)

stat_data_12 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data2, np.ones((data2.shape[0], 1))), axis = 1)), axis = 0)
stat_data_13 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data3, np.ones((data3.shape[0], 1))), axis = 1)), axis = 0)
stat_data_14 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data4, np.ones((data4.shape[0], 1))), axis = 1)), axis = 0)

stats = glmm.run(stat_data_12, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
print(np.asarray(stats))
stats = glmm.run(stat_data_13, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
print(np.asarray(stats))
stats = glmm.run(stat_data_14, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
print(np.asarray(stats))

axes[0, 0].set_title("Highly tremor, mostly burst")
axes[0, 1].set_title("Highly tremor, mostly non burst")
axes[1, 0].set_title("Little tremor, mostly burst")
axes[1, 1].set_title("Little tremor, mostly non burst")
plt.tight_layout()
plt.show(block = True)



