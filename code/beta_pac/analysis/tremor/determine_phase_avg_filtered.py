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

def get_values(path, subpath, mode, tremor_type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list()
    dir_list = list()
    f_names = list()
    
    patients = list()
    trials = list()
    
    #directionality = np.load("/mnt/data/Professional/UHN/pac_investigation/code/tremor_pac/analysis/tremor/test.npy")
    directionality = np.load("/mnt/data/Professional/UHN/pac_investigation/code/beta_pac/analysis/tremor/dac.npy")
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
        
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[4])), np.argmin(np.abs(data[7])))
        curr_dir = directionality[dir_names.index(f_name), [1, 2]]
        
#        loc_data = (np.argmax(data[0]), np.argmax(data[3]), np.argmax(data[6]))
        phase_shifts.append(loc_data)
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append([data[1], data[4], data[7]])
        sin_fit_list.append([data[0], data[3], data[6]])
        fit_list0.append(data[1]); fit_list1.append(data[4]); fit_list2.append(data[7])
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
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials), dir_list, f_names)

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, tremor_type, axes, axes2, dir_type, dir_thresh):
    (values, data, fit, patients, trials, dirs, f_names) = get_values(path, subpath, mode, tremor_type)
    print(values.shape)
    patients1 = np.copy(patients); patients2 = np.copy(patients)
    trials1 = np.copy(trials); trials2 = np.copy(trials)
    f_names1 = np.copy(f_names); f_names2 = np.copy(f_names)
    
    loc_burst_data = data[:, 1, :]; loc_burst_fit = fit[:, 1, :]
    loc_non_burst_data = data[:, 2, :]; loc_non_burst_fit = fit[:, 2, :]
    
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
    
    loc_burst_data = loc_burst_data[burst_dirs_idx, :]; patients1 = patients1[burst_dirs_idx]; trials1 = trials1[burst_dirs_idx]; loc_burst_fit = loc_burst_fit[burst_dirs_idx, :]; f_names1 = f_names1[burst_dirs_idx] 
    loc_non_burst_data = loc_non_burst_data[non_burst_dirs_idx, :]; patients2 = patients2[non_burst_dirs_idx]; trials2 = trials2[non_burst_dirs_idx]; loc_non_burst_fit = loc_non_burst_fit[non_burst_dirs_idx, :]; f_names2 = f_names2[non_burst_dirs_idx]

    
    pre = (loc_burst_data.shape[0], loc_non_burst_data.shape[0])
    
    #===========================================================================
    # loc_burst_data = out_rem.run(loc_burst_data, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # patients1 = out_rem.run(patients1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # trials1 = out_rem.run(trials1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # loc_non_burst_data = out_rem.run(loc_non_burst_data, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # patients2 = out_rem.run(patients2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    # trials2 = out_rem.run(trials2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    #===========================================================================
    
    print(mode, tremor_type, "%2.2f | %2.2f" % (np.float32(loc_burst_data.shape[0]/pre[0]), np.float32(loc_non_burst_data.shape[0]/pre[1]),))
    
    loc_burst_data_m = np.mean(loc_burst_data, axis = 0)
    loc_non_burst_data_m = np.mean(loc_non_burst_data, axis = 0)
    min_val = np.min([loc_burst_data_m, loc_non_burst_data_m]); loc_burst_data_m -= min_val; loc_non_burst_data_m -= min_val
    max_val = np.max([loc_burst_data_m, loc_non_burst_data_m]); loc_burst_data_m /= max_val; loc_non_burst_data_m /= max_val
    loc_burst_data_m -= 0.5; loc_burst_data_m *= 2; loc_non_burst_data_m -= 0.5; loc_non_burst_data_m *= 2
        
    tmp = 361
    ideal_slope = np.sin(2 * np.pi * 1 * np.arange(0, tmp)/tmp)
    
    #===========================================================================
    # sq_error_1 = np.sum(np.power((loc_burst_data - ideal_slope), 2), axis = 1)
    # loc_burst_data = orem.run(np.copy(loc_burst_data), sq_error_1, 2, 5, 0)
    # patients1 = orem.run(np.copy(patients1), sq_error_1, 2, 5, 0)
    # trials1 = orem.run(np.copy(trials1), sq_error_1, 2, 5, 0)
    # sq_error_2 = np.sum(np.power((loc_non_burst_data - ideal_slope), 2), axis = 1)
    # loc_non_burst_data = orem.run(np.copy(loc_non_burst_data), sq_error_2, 2, 5, 0)
    # patients2 = orem.run(np.copy(patients2), sq_error_2, 2, 5, 0)
    # trials2 = orem.run(np.copy(trials2), sq_error_2, 2, 5, 0)
    #===========================================================================
    
    sq_error_1 = np.sum(np.power((loc_burst_data - ideal_slope), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_non_burst_data - ideal_slope), 2), axis = 1)
    print(loc_burst_data.shape)
    print(loc_non_burst_data.shape)
     
    axes[0].plot(np.mean(loc_burst_fit, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(np.mean(loc_non_burst_fit, axis = 0), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[0].plot(loc_burst_data_m, color = "black", label = "avg. burst", zorder = 3)
    for x in range(loc_burst_data.shape[0]):
        axes[0].plot(loc_burst_data[x, :], color = "green", alpha = 0.25, zorder = 1)
    axes[1].plot(loc_non_burst_data_m, color = "black", label = "avg. non burst", zorder = 3)
    for x in range(loc_non_burst_data.shape[0]):
        axes[1].plot(loc_non_burst_data[x, :], color = "blue", alpha = 0.25, zorder = 1)
    
    axes[0].set_ylim((-2.5, 2.5))    
    axes[1].set_ylim((-2.5, 2.5))
    #axes[0].legend()
    #axes[1].legend()
    axes2[0].hist(np.argmax(loc_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    axes2[1].hist(np.argmax(loc_non_burst_fit, axis = 1).squeeze(), 12, (0, 360))
    
    print("all", np.mean(values[:, 0]), np.sqrt(np.var(values[:, 0])))
    print("burst", np.mean(values[:, 1]), np.sqrt(np.var(values[:, 1])))
    print("non burst", np.mean(values[:, 2]), np.sqrt(np.var(values[:, 2])))
    
    return (np.concatenate((np.expand_dims(sq_error_1, axis = 1), np.expand_dims(patients1, axis = 1), np.expand_dims(trials1, axis = 1)), axis = 1),
            np.concatenate((np.expand_dims(sq_error_2, axis = 1), np.expand_dims(patients2, axis = 1), np.expand_dims(trials2, axis = 1)), axis = 1))

dir_type = 1
dir_thresh = 0.1

(fig, axes) = plt.subplots(2, 2)
(fig2, axes2) = plt.subplots(2, 2)
(data1, data2) = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/2/", "tremor", "hf tremor", axes[0, :], axes2[0, :], dir_type, dir_thresh)
(data3, data4) = main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/2/", "tremor", "hf non tremor", axes[1, :], axes2[1, :], dir_type, dir_thresh)

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

axes[0, 0].set_title("Highly tremor, mostly burst"); axes2[0, 0].set_title("Highly tremor, mostly burst")
axes[0, 1].set_title("Highly tremor, mostly non burst"); axes2[0, 1].set_title("Highly tremor, mostly non burst")
axes[1, 0].set_title("Little tremor, mostly burst"); axes2[1, 0].set_title("Little tremor, mostly burst")
axes[1, 1].set_title("Little tremor, mostly non burst"); axes2[1, 1].set_title("Little tremor, mostly non burst")
plt.tight_layout()
plt.show(block = True)



