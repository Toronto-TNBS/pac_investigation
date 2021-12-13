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

def get_values(path, subpath, mode, beta_type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/projects/old/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list(); fit_list3 = list(); fit_list4 = list()
    dir_list = list()
    f_names = list()
    pac_scores = list()
    
    patients = list()
    trials = list()
    
    #directionality = np.load("/mnt/data/Professional/UHN/projects/old/pac_investigation/code/beta_pac/analysis/beta/test.npy")
    directionality = np.load("/mnt/data/Professional/UHN/projects/old/pac_investigation/code/beta_pac/analysis/beta/dac.npy")
    dir_names = directionality[:, -1].tolist()
    
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        
        if (f_name not in dir_names):
            continue
        
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (beta_type == "hf beta" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (beta_type == "hf non beta" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        #print(f_name, data[2])
                
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[10])), np.argmin(np.abs(data[13])), np.argmin(np.abs(data[19])), np.argmin(np.abs(data[22])))
        curr_dir = directionality[dir_names.index(f_name), [1, 2]]
        pac_scores.append([data[2], data[11], data[14], data[20], data[23]])
        
#        loc_data = (np.argmax(data[0]), np.argmax(data[3]), np.argmax(data[6]))
        phase_shifts.append(loc_data)
        patients.append(meta_info["patient_id"][f_idx])
        trials.append(meta_info["trial"][f_idx])
        data_list.append([data[1], data[10], data[13], data[19], data[22]])
        sin_fit_list.append([data[0], data[9], data[12], data[18], data[21]])
        fit_list0.append(data[1]); fit_list1.append(data[10]); fit_list2.append(data[13]); fit_list3.append(data[19]); fit_list4.append(data[22])
        dir_list.append(curr_dir)
        f_names.append(f_name)
        
        #=======================================================================
        # if (np.abs(np.argmax(data[3]) - 90) < 30):
        #     print(f_name, np.argmax(data[3]))
        # if (np.abs(np.argmax(data[3]) - 270) < 30):
        #     print(f_name, np.argmax(data[3]))
        #=======================================================================

    fit_list0 = np.asarray(fit_list0); fit_list1 = np.asarray(fit_list1); fit_list2 = np.asarray(fit_list2); fit_list3 = np.asarray(fit_list3)
    dir_list = np.asarray(dir_list)
    pac_scores = np.asarray(pac_scores)
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list), np.asarray(patients), np.asarray(trials), dir_list, f_names, pac_scores)

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, beta_type, axes, axes2, dir_type, dir_thresh, ideal_ref_slope = None):
    (values, data, fit, patients, trials, dirs, f_names, pac_scores) = get_values(path, subpath, mode, beta_type)
    print(values.shape)
    patients1 = np.copy(patients); patients2 = np.copy(patients); patients3 = np.copy(patients); patients4 = np.copy(patients)
    trials1 = np.copy(trials); trials2 = np.copy(trials); trials3 = np.copy(trials); trials4 = np.copy(trials)
    f_names1 = np.copy(f_names); f_names2 = np.copy(f_names); f_names3 = np.copy(f_names); f_names4 = np.copy(f_names)
    
    loc_burst_data1 = data[:, 1, :]; loc_burst_fit1 = fit[:, 1, :]
    loc_burst_data2 = data[:, 2, :]; loc_burst_fit2 = fit[:, 2, :]
    loc_non_burst_data1 = data[:, 3, :]; loc_non_burst_fit1 = fit[:, 3, :]
    loc_non_burst_data2 = data[:, 4, :]; loc_non_burst_fit2 = fit[:, 4, :]
    pac_scores1 = np.copy(pac_scores[:, 1]); pac_scores2 = np.copy(pac_scores[:, 2]); pac_scores3 = np.copy(pac_scores[:, 3]); pac_scores4 = np.copy(pac_scores[:, 4])
    
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
    
    loc_burst_data1 = loc_burst_data1[burst_dirs_idx, :]; patients1 = patients1[burst_dirs_idx]; trials1 = trials1[burst_dirs_idx]; loc_burst_fit1 = loc_burst_fit1[burst_dirs_idx, :]; f_names1 = f_names1[burst_dirs_idx]
    loc_burst_data2 = loc_burst_data2[burst_dirs_idx, :]; patients2 = patients2[burst_dirs_idx]; trials2 = trials2[burst_dirs_idx]; loc_burst_fit2 = loc_burst_fit2[burst_dirs_idx, :]; f_names2 = f_names2[burst_dirs_idx] 
    loc_non_burst_data1 = loc_non_burst_data1[non_burst_dirs_idx, :]; patients3 = patients3[non_burst_dirs_idx]; trials3 = trials3[non_burst_dirs_idx]; loc_non_burst_fit1 = loc_non_burst_fit1[non_burst_dirs_idx, :]; f_names3 = f_names3[non_burst_dirs_idx]
    loc_non_burst_data2 = loc_non_burst_data2[non_burst_dirs_idx, :]; patients4 = patients4[non_burst_dirs_idx]; trials4 = trials4[non_burst_dirs_idx]; loc_non_burst_fit2 = loc_non_burst_fit2[non_burst_dirs_idx, :]; f_names4 = f_names4[non_burst_dirs_idx]
    
    loc_burst_data_m1 = np.mean(loc_burst_data1, axis = 0)
    loc_burst_data_v1 = np.sqrt(np.var(loc_burst_data1, axis = 0))
    loc_burst_data_m2 = np.mean(loc_burst_data2, axis = 0)
    loc_burst_data_v2 = np.sqrt(np.var(loc_burst_data2, axis = 0))
    loc_non_burst_data_m1 = np.mean(loc_non_burst_data1, axis = 0)
    loc_non_burst_data_v1 = np.sqrt(np.var(loc_non_burst_data1, axis = 0))
    loc_non_burst_data_m2 = np.mean(loc_non_burst_data2, axis = 0)
    loc_non_burst_data_v2 = np.sqrt(np.var(loc_non_burst_data2, axis = 0))
    
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
    amplitude_signal = np.mean(loc_burst_data1, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = 0.9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result1 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    amplitude_signal = np.mean(loc_burst_data2, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = 0.9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result2 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    amplitude_signal = np.mean(loc_non_burst_data1, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = 0.9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result3 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
    amplitude_signal = np.mean(loc_non_burst_data2, axis = 0)
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 0.5, min = 0.9, max = 1.1, vary = True)
    #params.add("amp", value = 0.5, min = np.max(np.abs(amplitude_signal))*.9, max = np.max(np.abs(amplitude_signal))*1.1, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result4 = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, max_nfev = 300)
            
    axes[0].plot(result1.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(result2.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[2].plot(result3.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[3].plot(result4.best_fit, "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[0].plot(loc_burst_data_m1, color = "black", label = "avg. burst", zorder = 3)
    axes[0].fill_between(np.arange(0, loc_burst_data_m1.shape[0]),
                         loc_burst_data_m1 + loc_burst_data_v1, 
                         loc_burst_data_m1 - loc_burst_data_v1, color = "green", alpha = 0.25, zorder = 1)
    axes[1].plot(loc_burst_data_m2, color = "black", label = "avg. burst", zorder = 3)
    axes[1].fill_between(np.arange(0, loc_burst_data_m2.shape[0]),
                         loc_burst_data_m2 + loc_burst_data_v2, 
                         loc_burst_data_m2 - loc_burst_data_v2, color = "green", alpha = 0.25, zorder = 1)
    axes[2].plot(loc_non_burst_data_m1, color = "black", label = "avg. non burst", zorder = 3)
    axes[2].fill_between(np.arange(0, loc_non_burst_data_m1.shape[0]),
                         loc_non_burst_data_m1 + loc_non_burst_data_v1, 
                         loc_non_burst_data_m1 - loc_non_burst_data_v1, color = "blue", alpha = 0.25, zorder = 1)
    axes[3].plot(loc_non_burst_data_m2, color = "black", label = "avg. non burst", zorder = 3)
    axes[3].fill_between(np.arange(0, loc_non_burst_data_m2.shape[0]),
                         loc_non_burst_data_m2 + loc_non_burst_data_v2, 
                         loc_non_burst_data_m2 - loc_non_burst_data_v2, color = "blue", alpha = 0.25, zorder = 1)
    
    axes[0].set_ylim((-2.5, 2.5))
    axes[1].set_ylim((-2.5, 2.5))
    axes[2].set_ylim((-2.5, 2.5))
    axes[3].set_ylim((-2.5, 2.5))
    axes2[0].hist(np.argmax(loc_burst_fit1, axis = 1).squeeze(), 12, (0, 360))
    axes2[1].hist(np.argmax(loc_burst_fit2, axis = 1).squeeze(), 12, (0, 360))
    axes2[2].hist(np.argmax(loc_non_burst_fit1, axis = 1).squeeze(), 12, (0, 360))
    axes2[3].hist(np.argmax(loc_non_burst_fit2, axis = 1).squeeze(), 12, (0, 360))
    
    if (ideal_ref_slope is None):
        ideal_ref_slope = np.mean(loc_burst_data2, axis = 0)
    
    #------------ sq_error_1 = np.power((loc_burst_data_m - ideal_ref_slope), 2)
    #-------- sq_error_2 = np.power((loc_non_burst_data_m - ideal_ref_slope), 2)
    
    sq_error_1 = np.sum(np.power((loc_burst_data1 - ideal_ref_slope), 2), axis = 1)
    sq_error_2 = np.sum(np.power((loc_burst_data2 - ideal_ref_slope), 2), axis = 1)
    sq_error_3 = np.sum(np.power((loc_non_burst_data1 - ideal_ref_slope), 2), axis = 1)
    sq_error_4 = np.sum(np.power((loc_non_burst_data2 - ideal_ref_slope), 2), axis = 1)
    
    print("all", np.mean(values[:, 0]), np.sqrt(np.var(values[:, 0])))
    print("burst", np.mean(values[:, 1]), np.sqrt(np.var(values[:, 1])), np.mean(pac_scores1))
    print("burst", np.mean(values[:, 2]), np.sqrt(np.var(values[:, 2])), np.mean(pac_scores2))
    print("non burst", np.mean(values[:, 3]), np.sqrt(np.var(values[:, 3])), np.mean(pac_scores3))
    print("non burst", np.mean(values[:, 4]), np.sqrt(np.var(values[:, 4])), np.mean(pac_scores4))
    
    return (np.concatenate((np.expand_dims(sq_error_1, axis = 1), np.expand_dims(patients1, axis = 1), np.expand_dims(trials1, axis = 1)), axis = 1),
            np.concatenate((np.expand_dims(sq_error_2, axis = 1), np.expand_dims(patients2, axis = 1), np.expand_dims(trials2, axis = 1)), axis = 1), 
            np.concatenate((np.expand_dims(sq_error_3, axis = 1), np.expand_dims(patients3, axis = 1), np.expand_dims(trials3, axis = 1)), axis = 1), 
            np.concatenate((np.expand_dims(sq_error_4, axis = 1), np.expand_dims(patients4, axis = 1), np.expand_dims(trials4, axis = 1)), axis = 1), 
            ideal_ref_slope)

dir_type = 1
dir_thresh = 0.1

(fig, axes) = plt.subplots(2, 4, figsize = (int(156.962*4/100), int(85.678*3/100)), gridspec_kw = {'wspace':0, 'hspace':0})
(fig2, axes2) = plt.subplots(2, 4, figsize = (int(156.962*4/100), int(85.678*3/100)), gridspec_kw = {'wspace':0, 'hspace':0})
(fig3, axes3) = plt.subplots(2, 4, figsize = (int(156.962*4/100), int(85.678*3/100)), gridspec_kw = {'wspace':0, 'hspace':0})
(fig4, axes4) = plt.subplots(2, 4, figsize = (int(156.962*4/100), int(85.678*3/100)), gridspec_kw = {'wspace':0, 'hspace':0})

(data1, data2, data3, data4, loc_burst_fit)    = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf beta",
                                                      axes[0, :], axes2[0, :], dir_type, dir_thresh)
(data5, data6, data7, data8, _)                = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf non beta",
                                                      axes[1, :], axes2[1, :], dir_type, dir_thresh, loc_burst_fit)

dir_type = -1
dir_thresh = 0.1
(data9, data10, data11, data12, _)                = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf beta",
                                                         axes3[0, :], axes4[0, :], dir_type, dir_thresh, loc_burst_fit)
(data13, data14, data15, data16, _)             = main("/mnt/data/Professional/UHN/projects/old/pac_investigation/results/", "/data/2/", "beta", "hf non beta",
                                                       axes3[1, :], axes4[1, :], dir_type, dir_thresh, loc_burst_fit)


data1 = np.asarray(data1, dtype = np.float32);
data2 = np.asarray(data2, dtype = np.float32);
data3 = np.asarray(data3, dtype = np.float32);
data4 = np.asarray(data4, dtype = np.float32);
data5 = np.asarray(data5, dtype = np.float32);
data6 = np.asarray(data6, dtype = np.float32);
data7 = np.asarray(data7, dtype = np.float32);
data8 = np.asarray(data8, dtype = np.float32);
data9 = np.asarray(data9, dtype = np.float32);
data10= np.asarray(data10, dtype = np.float32);
data11= np.asarray(data11, dtype = np.float32);
data12= np.asarray(data12, dtype = np.float32);
data13= np.asarray(data13, dtype = np.float32);
data14= np.asarray(data14, dtype = np.float32);
data15= np.asarray(data15, dtype = np.float32);
data16= np.asarray(data16, dtype = np.float32);

stat_data_21  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data1,  np.ones((data1.shape[0], 1))), axis = 1)), axis = 0)
stat_data_23  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data3,  np.ones((data3.shape[0], 1))), axis = 1)), axis = 0)
stat_data_24  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data4,  np.ones((data4.shape[0], 1))), axis = 1)), axis = 0)
stat_data_25  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data5,  np.ones((data5.shape[0], 1))), axis = 1)), axis = 0)
stat_data_26  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data6,  np.ones((data6.shape[0], 1))), axis = 1)), axis = 0)
stat_data_27  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data7,  np.ones((data7.shape[0], 1))), axis = 1)), axis = 0)
stat_data_28  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data8,  np.ones((data8.shape[0], 1))), axis = 1)), axis = 0)
stat_data_29  = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data9,  np.ones((data9.shape[0], 1))), axis = 1)), axis = 0)
stat_data_210 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data10, np.ones((data10.shape[0], 1))), axis = 1)), axis = 0)
stat_data_211 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data11, np.ones((data11.shape[0], 1))), axis = 1)), axis = 0)
stat_data_212 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data12, np.ones((data12.shape[0], 1))), axis = 1)), axis = 0)
stat_data_213 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data13, np.ones((data13.shape[0], 1))), axis = 1)), axis = 0)
stat_data_214 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data14, np.ones((data14.shape[0], 1))), axis = 1)), axis = 0)
stat_data_215 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data15, np.ones((data15.shape[0], 1))), axis = 1)), axis = 0)
stat_data_216 = np.concatenate((np.concatenate((data2, np.zeros((data2.shape[0], 1))), axis = 1), np.concatenate((data16, np.ones((data16.shape[0], 1))), axis = 1)), axis = 0)

print(dir_type, dir_thresh)
vals = list()

stats = glmm.run(stat_data_21, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_23, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_24, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_25, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_26, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_27, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_28, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_29, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_210, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_211, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_212, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_213, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_214, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_215, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])
stats = glmm.run(stat_data_216, ["score", "patient", "trial", "type"], ["continuous", "categorical", "categorical", "categorical"],
                 "score ~ type + (1|patient) + (1|trial)", "list(score = contr.sum, patient = contr.sum, trial = contr.sum, type = contr.sum)",
                 "gaussian")
stats = np.asarray(stats)
print(stats)
print(float(stats[2, 0])*15, "%05.03f, %05.03f, %05.03f" % ((float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]), (float(stats[3, 0]) + float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1])))
vals.append([(float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]), np.abs((float(stats[3, 0]) - float(stats[4, 0]) + float(stats[3, 1]))/float(stats[3, 1]) - (float(stats[3, 0]) + float(stats[3, 1]))/float(stats[3, 1]))])

plt.figure()
plt.errorbar(np.arange(0, 15), np.asarray(vals)[:, 0], yerr = np.asarray(vals)[:, 1])
plt.bar(np.arange(0, 15), np.asarray(vals)[:, 0])

#---------------------------------- stats = scipy.stats.ttest_1samp(data3, 0)[1]
# print(stats, np.mean(data3), np.mean(data3) - np.sqrt(np.var(data3)), np.mean(data3) + np.sqrt(np.var(data3)))
#---------------------------------- stats = scipy.stats.ttest_1samp(data5, 0)[1]
# print(stats, np.mean(data5), np.mean(data5) - np.sqrt(np.var(data5)), np.mean(data5) + np.sqrt(np.var(data5)))
#---------------------------------- stats = scipy.stats.ttest_1samp(data7, 0)[1]
# print(stats, np.mean(data7), np.mean(data7) - np.sqrt(np.var(data7)), np.mean(data7) + np.sqrt(np.var(data7)))

#===============================================================================
# axes[0, 0].set_title("Highly beta, mostly burst"); axes2[0, 0].set_title("Highly beta, mostly burst")
# axes[0, 1].set_title("Highly beta, mostly non burst"); axes2[0, 1].set_title("Highly beta, mostly non burst")
# axes[1, 0].set_title("Little beta, mostly burst"); axes2[1, 0].set_title("Little beta, mostly burst")
# axes[1, 1].set_title("Little beta, mostly non burst"); axes2[1, 1].set_title("Little beta, mostly non burst")
#===============================================================================
plt.tight_layout()
plt.show(block = True)



