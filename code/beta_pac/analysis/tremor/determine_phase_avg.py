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

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    meta_info["file"]
    
    phase_shifts = list()
    data_list = list()
    sin_fit_list = list()
    fit_list0 = list(); fit_list1 = list(); fit_list2 = list()
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (type == "hf tremor" and int(int(float(meta_info["hf auto"][f_idx]))) == 0):
            continue
        if (type == "hf non tremor" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        
        
        #print(f_name, data[2])
        
        loc_data = (np.argmin(np.abs(data[1])), np.argmin(np.abs(data[4])), np.argmin(np.abs(data[7])))
        
#        loc_data = (np.argmax(data[0]), np.argmax(data[3]), np.argmax(data[6]))
        phase_shifts.append(loc_data)
        data_list.append([data[1], data[4], data[7]])
        sin_fit_list.append([data[0], data[3], data[6]])
        fit_list0.append(data[1]); fit_list1.append(data[4]); fit_list2.append(data[7])

    fit_list0 = np.asarray(fit_list0); fit_list1 = np.asarray(fit_list1); fit_list2 = np.asarray(fit_list2)
       
    return (np.asarray(phase_shifts), np.asarray(data_list), np.asarray(sin_fit_list))

import finn.cleansing.outlier_removal as out_rem

def main(path, subpath, mode, type, axes):
    (values, data, fit) = get_values(path, subpath, mode, type)
    
    loc_data1 = data[:, 1, :]; loc_fit1 = fit[:, 1, :]
    loc_data2 = data[:, 2, :]; loc_fit2 = fit[:, 2, :]
    
    pre = (loc_data1.shape[0], loc_data2.shape[0])
    loc_data1 = out_rem.run(loc_data1, np.argmax(loc_fit1, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    loc_data2 = out_rem.run(loc_data2, np.argmax(loc_fit2, axis = 1), max_std_dist = 2, min_samp_cnt = 5, axis = 0)
    print("%2.2f | %2.2f" % (np.float32(loc_data1.shape[0]/pre[0]), np.float32(loc_data2.shape[0]/pre[1]),))
    
    loc_data1_m = np.mean(loc_data1, axis = 0)
    loc_data2_m = np.mean(loc_data2, axis = 0)
    min_val = np.min([loc_data1_m, loc_data2_m]); loc_data1_m -= min_val; loc_data2_m -= min_val
    max_val = np.max([loc_data1_m, loc_data2_m]); loc_data1_m /= max_val; loc_data2_m /= max_val
    loc_data1_m -= 0.5; loc_data1_m *= 2; loc_data2_m -= 0.5; loc_data2_m *= 2
        
    tmp = 361
    axes[0].plot(np.sin(2 * np.pi * 1 * np.arange(0, tmp)/tmp), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
    axes[1].plot(np.sin(2 * np.pi * 1 * np.arange(0, tmp)/tmp), "--", color = "red", zorder = 2, alpha = 0.5, label = "lfp")
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

(fig, axes) = plt.subplots(2, 2)
main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/2/", "tremor", "hf tremor", axes[0, :])
main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/2/", "tremor", "hf non tremor", axes[1, :])
axes[0, 0].set_title("Highly tremor, mostly burst")
axes[0, 1].set_title("Highly tremor, mostly non burst")
axes[1, 0].set_title("Little tremor, mostly burst")
axes[1, 1].set_title("Little tremor, mostly non burst")
plt.tight_layout()
plt.show(block = True)



