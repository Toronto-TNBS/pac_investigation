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

def get_values(path, subpath, mode, tremor_type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/projects/old/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list()
    dir_list = list()
    f_names = list()
    pac_scores = list()
    
    patients = list()
    trials = list()
    
    #directionality = np.load("/mnt/data/Professional/UHN/projects/old/pac_investigation/code/beta_pac/analysis/beta/test.npy")
    directionality = np.load("/mnt/data/Professional/UHN/projects/old/pac_investigation/code/beta_pac/analysis/tremor/dac.npy")
    dir_names = directionality[:, -1].tolist()
    
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        
        if (f_name not in dir_names):
            continue
        
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (tremor_type == "hf tremor" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (tremor_type == "hf non tremor" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        #print(f_name, data[2])
        
        offset = 0#6
        
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[7])), np.argmin(np.abs(data[16])))
        curr_dir = directionality[dir_names.index(f_name), [1, 2]]
        pac_scores.append([data[2], data[8], data[17]])
        
        phase_shifts.append(loc_data)
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append([data[1], data[7], data[16]])
        sin_fit_list.append([data[0], data[6], data[15]])
        fit_list0.append(data[1]); fit_list1.append(data[7]); fit_list2.append(data[16])
        dir_list.append(curr_dir)
        f_names.append(f_name)
        #=======================================================================
        # if (np.abs(np.argmax(data[3]) - 90) < 30):
        #     print(f_name, np.argmax(data[3]))
        # if (np.abs(np.argmax(data[3]) - 270) < 30):
        #     print(f_name, np.argmax(data[3]))
        #=======================================================================

    fit_list0 = np.asarray(fit_list0); fit_list1 = np.asarray(fit_list1); fit_list2 = np.asarray(fit_list2)
    dir_list = np.asarray(dir_list)
    pac_scores = np.asarray(pac_scores)
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials), dir_list, f_names, pac_scores)

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, tremor_type, axes, axes2, dir_type, dir_thresh, ideal_ref_slope = None):
    (values, data, fit, patients, trials, dirs, f_names, pac_scores) = get_values(path, subpath, mode, tremor_type)
    print(values.shape)
    patients1 = np.copy(patients); patients2 = np.copy(patients)
    trials1 = np.copy(trials); trials2 = np.copy(trials)
    f_names1 = np.copy(f_names); f_names2 = np.copy(f_names)
    
    loc_burst_data = data[:, 1, :]; loc_burst_fit = fit[:, 1, :]
    loc_non_burst_data = data[:, 2, :]; loc_non_burst_fit = fit[:, 2, :]
    pac_scores1 = np.copy(pac_scores[:, 1]); pac_scores2 = np.copy(pac_scores[:, 2])
    
    #filter data
    burst_dirs = np.asarray(np.copy(dirs[:, 0]), dtype = float)
    burst_dirs_idx_1 = np.argwhere(np.abs(np.asarray(burst_dirs, dtype = float)) > dir_thresh).squeeze()
    burst_dirs_idx_2 = np.argwhere(np.sign(burst_dirs[burst_dirs_idx_1]) == dir_type).squeeze()
    burst_dirs_idx = burst_dirs_idx_1[burst_dirs_idx_2]
    #burst_dirs = burst_dirs[burst_dirs_idx_1[burst_dirs_idx_2]]
    
    non_burst_dirs = np.asarray(np.copy(dirs[:, 0]), dtype = float)
    non_burst_dirs_idx_1 = np.argwhere(np.abs(np.asarray(non_burst_dirs, dtype = float)) > dir_thresh).squeeze()
    non_burst_dirs_idx_2 = np.argwhere(np.sign(non_burst_dirs[non_burst_dirs_idx_1]) == dir_type).squeeze()
    non_burst_dirs_idx = non_burst_dirs_idx_1[non_burst_dirs_idx_2]
    
    loc_burst_data = loc_burst_data[burst_dirs_idx, :]; patients1 = patients1[burst_dirs_idx]; trials1 = trials1[burst_dirs_idx]; loc_burst_fit = loc_burst_fit[burst_dirs_idx, :]; f_names1 = f_names1[burst_dirs_idx]; pac_scores1 = pac_scores1[burst_dirs_idx]
    loc_non_burst_data = loc_non_burst_data[non_burst_dirs_idx, :]; patients2 = patients2[non_burst_dirs_idx]; trials2 = trials2[non_burst_dirs_idx]; loc_non_burst_fit = loc_non_burst_fit[non_burst_dirs_idx, :]; f_names2 = f_names2[non_burst_dirs_idx]; pac_scores2 = pac_scores2[non_burst_dirs_idx]

    
    pre = (loc_burst_data.shape[0], loc_non_burst_data.shape[0])
    
    print(mode, tremor_type, "%2.2f | %2.2f" % (np.float32(loc_burst_data.shape[0]/pre[0]), np.float32(loc_non_burst_data.shape[0]/pre[1]),))
    
    loc_burst_data_m = np.mean(loc_burst_data, axis = 0)
    loc_burst_data_v = np.sqrt(np.var(loc_burst_data, axis = 0))
    loc_non_burst_data_m = np.mean(loc_non_burst_data, axis = 0)
    loc_non_burst_data_v = np.sqrt(np.var(loc_non_burst_data, axis = 0))
    #===========================================================================
    # min_val = np.min([loc_burst_data_m, loc_non_burst_data_m]); loc_burst_data_m -= min_val; loc_non_burst_data_m -= min_val
    # max_val = np.max([loc_burst_data_m, loc_non_burst_data_m]); loc_burst_data_m /= max_val; loc_non_burst_data_m /= max_val
    # loc_burst_data_m -= 0.5; loc_burst_data_m *= 2; loc_non_burst_data_m -= 0.5; loc_non_burst_data_m *= 2
    #===========================================================================
    
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
    amplitude_signal = np.mean(loc_burst_data, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    params.add("amp", value = 0.5, min = .9, max = 1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result1 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    amplitude_signal = np.mean(loc_non_burst_data, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    params.add("amp", value = 0.5, min = .9, max = 1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result2 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    
    axes[0].plot(result1.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(result2.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[0].plot(loc_burst_data_m, color = "black", label = "avg. burst", zorder = 3)
    #---------------------------------- for x in range(loc_burst_data.shape[0]):
        # axes[0].plot(loc_burst_data[x, :], color = "green", alpha = 0.25, zorder = 1)
    axes[0].fill_between(np.arange(0, loc_burst_data_m.shape[0]),
                         loc_burst_data_m + loc_burst_data_v, 
                         loc_burst_data_m - loc_burst_data_v, color = "green", alpha = 0.25, zorder = 1)
    axes[1].plot(loc_non_burst_data_m, color = "black", label = "avg. non burst", zorder = 3)
    #------------------------------ for x in range(loc_non_burst_data.shape[0]):
        # axes[1].plot(loc_non_burst_data[x, :], color = "blue", alpha = 0.25, zorder = 1)
    axes[1].fill_between(np.arange(0, loc_non_burst_data_m.shape[0]),
                         loc_non_burst_data_m + loc_non_burst_data_v, 
                         loc_non_burst_data_m - loc_non_burst_data_v, color = "blue", alpha = 0.25, zorder = 1)
    
    axes[0].set_ylim((-2.5, 2.5))    
    axes[1].set_ylim((-2.5, 2.5))
    axes2[0].hist(np.argmax(loc_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    axes2[1].hist(np.argmax(loc_non_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    
    if (ideal_ref_slope is None):
        ideal_ref_slope = np.mean(loc_burst_data, axis = 0)
    
    #------------ sq_error_1 = np.power((loc_burst_data_m - ideal_ref_slope), 2)
    #-------- sq_error_2 = np.power((loc_non_burst_data_m - ideal_ref_slope), 2)
    
    sq_error_1 = np.sum(np.power((loc_burst_data - ideal_ref_slope), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_non_burst_data - ideal_ref_slope), 2), axis = 1)
    
    sq_error_1 = np.sum(np.power((loc_burst_data - result1.best_fit), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_non_burst_data - result2.best_fit), 2), axis = 1)
    
    #===========================================================================
    # std_dev = 1.85
    # print(loc_burst_data.shape)
    # print(loc_non_burst_data.shape)
    # loc_burst_data = orem.run(np.copy(loc_burst_data), sq_error_1, std_dev, 5, 0)
    # patients1 = orem.run(np.copy(patients1), sq_error_1, std_dev, 5, 0)
    # trials1 = orem.run(np.copy(trials1), sq_error_1, std_dev, 5, 0)
    # sq_error_1 = orem.run(sq_error_1, sq_error_1, std_dev, 5, 0)
    # loc_non_burst_data = orem.run(np.copy(loc_non_burst_data), sq_error_2, std_dev, 5, 0)
    # patients2 = orem.run(np.copy(patients2), sq_error_2, std_dev, 5, 0)
    # trials2 = orem.run(np.copy(trials2), sq_error_2, std_dev, 5, 0)
    # sq_error_2 = orem.run(sq_error_2, sq_error_2, std_dev, 5, 0)
    # print(loc_burst_data.shape)
    # print(loc_non_burst_data.shape)
    #===========================================================================
    
    print("all", np.mean(values[:, 0]), np.sqrt(np.var(values[:, 0])))
    print("burst", np.mean(values[:, 1]), np.sqrt(np.var(values[:, 1])), np.mean(pac_scores1))
    print("non burst", np.mean(values[:, 2]), np.sqrt(np.var(values[:, 2])), np.mean(pac_scores2))
         
    #===========================================================================
    # axes[0].plot(np.mean(loc_burst_fit, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    # axes[1].plot(np.mean(loc_non_burst_fit, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    # axes[0].plot(loc_burst_data_m, color = "black", label = "avg. burst", zorder = 3)
    # for x in range(loc_burst_data.shape[0]):
    #     axes[0].plot(loc_burst_data[x, :], color = "green", alpha = 0.25, zorder = 1)
    # axes[1].plot(loc_non_burst_data_m, color = "black", label = "avg. non burst", zorder = 3)
    # for x in range(loc_non_burst_data.shape[0]):
    #     axes[1].plot(loc_non_burst_data[x, :], color = "blue", alpha = 0.25, zorder = 1)
    # 
    # axes[0].set_ylim((-2.5, 2.5))    
    # axes[1].set_ylim((-2.5, 2.5))
    # axes2[0].hist(np.argmax(loc_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    # axes2[1].hist(np.argmax(loc_non_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    #===========================================================================
    
    return (np.concatenate((np.expand_dims(pac_scores1, axis = 1), np.expand_dims(sq_error_1, axis = 1), np.expand_dims(patients1, axis = 1), np.expand_dims(trials1, axis = 1)), axis = 1),
            np.concatenate((np.expand_dims(pac_scores2, axis = 1), np.expand_dims(sq_error_2, axis = 1), np.expand_dims(patients2, axis = 1), np.expand_dims(trials2, axis = 1)), axis = 1), 
            ideal_ref_slope)
    
    return (np.expand_dims(sq_error_1, axis = 1), np.expand_dims(sq_error_2, axis = 1), 
            ideal_ref_slope)

dir_type = -1
dir_thresh = 0.0

(fig, axes) = plt.subplots(2, 2)
(fig2, axes2) = plt.subplots(2, 2)
(fig3, axes3) = plt.subplots(2, 2)
(fig4, axes4) = plt.subplots(2, 2)

(data1, data2, loc_burst_fit)   = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "tremor", "hf tremor",
                                       axes[0, :], axes2[0, :], dir_type, dir_thresh)
(data3, data4, _)               = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "tremor", "hf non tremor",
                                       axes[1, :], axes2[1, :], dir_type, dir_thresh, loc_burst_fit)

dir_type = 1
dir_thresh = 0.0
(data5, data6, _)                = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "tremor", "hf tremor",
                                       axes3[0, :], axes4[0, :], dir_type, dir_thresh, loc_burst_fit)
(data7, data8, _)                = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "tremor", "hf non tremor",
                                       axes3[1, :], axes4[1, :], dir_type, dir_thresh, loc_burst_fit)


data1 = np.asarray(data1, dtype = np.float32);
data3 = np.asarray(data3, dtype = np.float32);
data5 = np.asarray(data5, dtype = np.float32);
data7 = np.asarray(data7, dtype = np.float32);

stat_data_12 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data3, np.ones((data3.shape[0], 1))), axis = 1)), axis = 0)
stat_data_13 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data5, np.ones((data5.shape[0], 1))), axis = 1)), axis = 0)
stat_data_14 = np.concatenate((np.concatenate((data1, np.zeros((data1.shape[0], 1))), axis = 1), np.concatenate((data7, np.ones((data7.shape[0], 1))), axis = 1)), axis = 0)
 
print(dir_type, dir_thresh)
 
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

#---------------------------------- stats = scipy.stats.ttest_1samp(data3, 0)[1]
# print(stats, np.mean(data3), np.mean(data3) - np.sqrt(np.var(data3)), np.mean(data3) + np.sqrt(np.var(data3)))
#---------------------------------- stats = scipy.stats.ttest_1samp(data5, 0)[1]
# print(stats, np.mean(data5), np.mean(data5) - np.sqrt(np.var(data5)), np.mean(data5) + np.sqrt(np.var(data5)))
#---------------------------------- stats = scipy.stats.ttest_1samp(data7, 0)[1]
# print(stats, np.mean(data7), np.mean(data7) - np.sqrt(np.var(data7)), np.mean(data7) + np.sqrt(np.var(data7)))

axes[0, 0].set_title("Highly tremor, mostly burst"); axes2[0, 0].set_title("Highly tremor, mostly burst")
axes[0, 1].set_title("Highly tremor, mostly non burst"); axes2[0, 1].set_title("Highly tremor, mostly non burst")
axes[1, 0].set_title("Little tremor, mostly burst"); axes2[1, 0].set_title("Little tremor, mostly burst")
axes[1, 1].set_title("Little tremor, mostly non burst"); axes2[1, 1].set_title("Little tremor, mostly non burst")
plt.tight_layout()
plt.show(block = True)



