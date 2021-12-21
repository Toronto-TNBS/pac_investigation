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

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/projects/old/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list()
    
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
        
        
        #print(f_name, data[2])
        
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[7])), np.argmin(np.abs(data[16])))
        pac_scores.append([data[2], data[8], data[17]])
        
#        loc_data = (np.argmax(data[0]), np.argmax(data[3]), np.argmax(data[6]))
        phase_shifts.append(loc_data)
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append([data[1], data[7], data[16]])
        sin_fit_list.append([data[0], data[6], data[15]])
        fit_list0.append(data[1]); fit_list1.append(data[7]); fit_list2.append(data[16])

    fit_list0 = np.asarray(fit_list0); fit_list1 = np.asarray(fit_list1); fit_list2 = np.asarray(fit_list2)
    pac_scores = np.asarray(pac_scores)
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials), pac_scores)

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, type, axes, ideal_ref_slope = None):
    (values, data, fit, patients, trials, pac_scores) = get_values(path, subpath, mode, type)
    patients1 = np.copy(patients); patients2 = np.copy(patients)
    trials1 = np.copy(trials); trials2 = np.copy(trials)
    
    loc_data1 = data[:, 1, :]; loc_fit1 = fit[:, 1, :]
    loc_data2 = data[:, 2, :]; loc_fit2 = fit[:, 2, :]
    pac_scores1 = np.copy(pac_scores[:, 1]); pac_scores2 = np.copy(pac_scores[:, 2])
    
    pre = (loc_data1.shape[0], loc_data2.shape[0])
    
    #===========================================================================
    # loc_data1 = out_rem.run(loc_data1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # patients1 = out_rem.run(patients1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # trials1 = out_rem.run(trials1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # loc_data2 = out_rem.run(loc_data2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # patients2 = out_rem.run(patients2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # trials2 = out_rem.run(trials2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    #===========================================================================
    
    print(mode, type, "%2.2f | %2.2f" % (np.float32(loc_data1.shape[0]/pre[0]), np.float32(loc_data2.shape[0]/pre[1]),))
    
    loc_data1_m = np.mean(loc_data1, axis = 0)
    loc_data2_m = np.mean(loc_data2, axis = 0)
    loc_data1_v = np.sqrt(np.var(loc_data1, axis = 0))
    loc_data2_v = np.sqrt(np.var(loc_data2, axis = 0))

    # min_val = np.min([loc_data1_m, loc_data2_m]); loc_data1_m -= min_val; loc_data1_v -= min_val; loc_data2_m -= min_val; loc_data2_v -= min_val
    # max_val = np.max([loc_data1_m, loc_data2_m]); loc_data1_m /= max_val; loc_data1_v /= max_val; loc_data2_m /= max_val
    # loc_data1_m -= 0.5; loc_data1_m *= 2; loc_data2_m -= 0.5; loc_data2_m *= 2
    
        
    if (ideal_ref_slope is None):
        ideal_ref_slope = np.mean(loc_data1, axis = 0)
    
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
    amplitude_signal = np.mean(loc_data1, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = .9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result1 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    amplitude_signal = np.mean(loc_data2, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = .9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result2 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
             
    axes[0].plot(result1.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(result2.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[0].plot(loc_data1_m, color = "black", label = "avg. burst", zorder = 3)
    #--------------------------------------- for x in range(loc_data1.shape[0]):
        # axes[0].plot(loc_data1[x, :], color = "green", alpha = 0.25, zorder = 1)
    axes[0].fill_between(np.arange(0, loc_data1.shape[1]),
                         loc_data1_m + loc_data1_v, 
                         loc_data1_m - loc_data1_v, color = "green", alpha = 0.25, zorder = 1)
    
    axes[1].plot(loc_data2_m, color = "black", label = "avg. non burst", zorder = 3)
    #--------------------------------------- for x in range(loc_data2.shape[0]):
        # axes[1].plot(loc_data2[x, :], color = "blue", alpha = 0.25, zorder = 1)
    axes[1].fill_between(np.arange(0, loc_data2.shape[1]),
                         loc_data2_m + loc_data2_v, 
                         loc_data2_m - loc_data2_v, color = "blue", alpha = 0.25, zorder = 1)
    
    axes[0].set_ylim((-2.5, 2.5))    
    axes[1].set_ylim((-2.5, 2.5))
    #axes[0].legend()
    #axes[1].legend()
    
    #----------------- sq_error_1 = np.power((loc_data1_m - ideal_ref_slope), 2)
    #----------------- sq_error_2 = np.power((loc_data2_m - ideal_ref_slope), 2)
    
    sq_error_1 = np.sum(np.power((loc_data1 - ideal_ref_slope), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_data2 - ideal_ref_slope), 2), axis = 1)
    
    sq_error_1 = np.sum(np.power((loc_data1 - result1.best_fit), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_data2 - result2.best_fit), 2), axis = 1)
    
#===============================================================================
#     std_dev = 1.85
#     print(loc_data1.shape)
#     print(loc_data2.shape)
#     sq_error_1 = np.sum(np.power((loc_data1 - ideal_ref_slope), 2), axis = 1)
#     loc_data1 = orem.run(np.copy(loc_data1), sq_error_1, std_dev, 5, 0)
#     patients1 = orem.run(np.copy(patients1), sq_error_1, std_dev, 5, 0)
#     trials1 = orem.run(np.copy(trials1), sq_error_1, std_dev, 5, 0)
#     sq_error_1 = orem.run(sq_error_1, sq_error_1, std_dev, 5, 0)
#     sq_error_2 = np.sum(np.power((loc_data2 - ideal_ref_slope), 2), axis = 1)
#     loc_data2 = orem.run(np.copy(loc_data2), sq_error_2, std_dev, 5, 0)
#     patients2 = orem.run(np.copy(patients2), sq_error_2, std_dev, 5, 0)
#     trials2 = orem.run(np.copy(trials2), sq_error_2, std_dev, 5, 0)
#     sq_error_2 = orem.run(sq_error_2, sq_error_2, std_dev, 5, 0)
# 
#     print(loc_data1.shape)
#     print(loc_data2.shape)
#===============================================================================
     
    #===========================================================================
    # axes[0].plot(np.mean(loc_fit1, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    # axes[1].plot(np.mean(loc_fit2, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    # axes[0].plot(loc_data1_m, color = "black", label = "avg. burst", zorder = 3)
    # for x in range(loc_data1.shape[0]):
    #     axes[0].plot(loc_data1[x, :], color = "green", alpha = 0.25, zorder = 1)
    # axes[1].plot(loc_data2_m, color = "black", label = "avg. non burst", zorder = 3)
    # for x in range(loc_data2.shape[0]):
    #     axes[1].plot(loc_data2[x, :], color = "blue", alpha = 0.25, zorder = 1)
    # 
    # axes[0].set_ylim((-2.5, 2.5))    
    # axes[1].set_ylim((-2.5, 2.5))
    # #axes[0].legend()
    # #axes[1].legend()
    #===========================================================================
    
    print("all", np.mean(values[:, 0]), np.sqrt(np.var(values[:, 0])))
    print("burst", np.mean(values[:, 1]), np.sqrt(np.var(values[:, 1])))
    print("non burst", np.mean(values[:, 2]), np.sqrt(np.var(values[:, 2])))
    
    return (np.concatenate((np.expand_dims(pac_scores1, axis = 1), np.expand_dims(sq_error_1, axis = 1), np.expand_dims(patients1, axis = 1), np.expand_dims(trials1, axis = 1)), axis = 1),
            np.concatenate((np.expand_dims(pac_scores2, axis = 1), np.expand_dims(sq_error_2, axis = 1), np.expand_dims(patients2, axis = 1), np.expand_dims(trials2, axis = 1)), axis = 1), 
            ideal_ref_slope)
    
    return (np.expand_dims(sq_error_1, axis = 1), np.expand_dims(sq_error_2, axis = 1),
            ideal_ref_slope)

(fig, axes) = plt.subplots(2, 2)
(data1, data2, loc_burst_fit)   = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf beta", axes[0, :])
(data3, data4, _)               = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf non beta", axes[1, :], loc_burst_fit)

data1 = np.asarray(data1, dtype = np.float32);
data2 = np.asarray(data2, dtype = np.float32);
data3 = np.asarray(data3, dtype = np.float32);
data4 = np.asarray(data4, dtype = np.float32);

stat_data_12 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data2, np.ones((data2.shape[0], 1))), axis = 1)), axis = 0)
stat_data_13 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data3, np.ones((data3.shape[0], 1))), axis = 1)), axis = 0)
stat_data_14 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data4, np.ones((data4.shape[0], 1))), axis = 1)), axis = 0)
 
stats = glmm.run(stat_data_12, ["pac_score", "shape_score", "patient", "trial", "type"], ["continuous", "continuous", "categorical", "categorical", "categorical"],
                 "pac_score ~ type + (1|patient) + (1|trial)", "list(pac_score = contr.sum, shape_score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
#print(stats)
print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])), float(stats[3, 0]) + float(stats[3, 1]), float(stats[3, 1]))
stats = glmm.run(stat_data_13, ["pac_score", "shape_score", "patient", "trial", "type"], ["continuous", "continuous", "categorical", "categorical", "categorical"],
                 "pac_score ~ type + (1|patient) + (1|trial)", "list(pac_score = contr.sum, shape_score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
#print(stats)
print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])), float(stats[3, 0]) + float(stats[3, 1]), float(stats[3, 1]))
stats = glmm.run(stat_data_14, ["pac_score", "shape_score", "patient", "trial", "type"], ["continuous", "continuous", "categorical", "categorical", "categorical"],
                 "pac_score ~ type + (1|patient) + (1|trial)", "list(pac_score = contr.sum, shape_score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
#print(stats)
print(float(stats[2, 0])*3, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])), float(stats[3, 0]) + float(stats[3, 1]), float(stats[3, 1]))

#---------------------------------- stats = scipy.stats.ttest_1samp(data2, 0)[1]
# print(stats, np.mean(data2), np.mean(data2) - np.var(data2), np.mean(data2) + np.var(data2))
#---------------------------------- stats = scipy.stats.ttest_1samp(data3, 0)[1]
# print(stats, np.mean(data3), np.mean(data3) - np.var(data3), np.mean(data3) + np.var(data3))
#---------------------------------- stats = scipy.stats.ttest_1samp(data4, 0)[1]
# print(stats, np.mean(data4), np.mean(data4) - np.var(data4), np.mean(data4) + np.var(data4))

axes[0, 0].set_title("Highly beta, mostly burst")
axes[0, 1].set_title("Highly beta, mostly non burst")
axes[1, 0].set_title("Little beta, mostly burst")
axes[1, 1].set_title("Little beta, mostly non burst")
plt.tight_layout()
plt.show(block = True)



