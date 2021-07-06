'''
Created on Jun 10, 2021

@author: voodoocode
'''


import pickle
import numpy as np
import methods.data_io.ods

import matplotlib
import lmfit
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.stats

def get_values(path, subpath, mode, type):
    meta_data = methods.data_io.ods.ods_data("/mnt/data/Professional/UHN/pac_investigation/data/meta.ods")
    meta_info = meta_data.get_sheet_as_dict(mode)
    
    f_names = list()
    data = list()
    for (f_idx, f_name) in enumerate(meta_info["file"]):
        if (int(meta_info["valid_data"][f_idx]) == 0):
            continue
        
        if (type == "beta" and int(float(meta_info["hf auto"][f_idx])) == 0):
            continue
        if (type == "non beta" and int(float(meta_info["hf auto"][f_idx])) == 1):
            continue
                
        loc_data = pickle.load(open(path + mode + subpath + f_name + ".pkl", "rb"))
        data.append(loc_data)
        f_names.append(f_name)
       
    return (np.asarray(data), f_names)

import finn.cleansing.outlier_removal as out_rem

def __gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-.5 * np.power((x - mu)/sigma, 2))

def __triangle(x, center):
    tmp = np.zeros((len(x)))
    tmp[(center - 1):(center + 1)] += 0.5
    tmp[center] += .5
    
    return tmp

def main(path, subpath, mode, type = "beta"):
    (data, f_names) = get_values(path, subpath, mode, type)
    for idx in range(data.shape[0]):
        data[idx, :] = np.convolve(np.pad(data[idx, :], 2, "constant", constant_values = 0), np.ones(5)/5, "valid")
    data = data[:, 2:-2]
    
    scores = list()
    params = list()
    for idx in range(data.shape[0]):
        param = lmfit.Parameters()
        param.add("mu", value = 0, min = 0, max = 14, vary = True)
        param.add("sigma", value = 1, min = 0, max = 2, vary = True)
        model = lmfit.Model(__gaussian, nan_policy = "omit")
        result = model.fit(data[idx, :], x = np.arange(0, len(data[idx, :]), 1), params = param)
        score = np.sum(np.power(result.best_fit - data[idx, :], 2))
        scores.append(score)
        params.append(result.best_values)
    scores = np.asarray(scores)
    params = np.asarray(params)
    
    for idx in range(data.shape[0]):
        plt.figure()
        plt.plot(data[idx])
        if (type == "beta"):
            plt.savefig(path + mode + "/img/5/beta/" + f_names[idx] + ".png")
        else:
            plt.savefig(path + mode + "/img/5/non_beta/" + f_names[idx] + ".png")
        plt.close()
    
    #Plot stuff shifted
    centered_data = np.zeros(100)
    for idx in range(data.shape[0]):
        ref_idx = 50 - int(params[idx]["mu"])
        centered_data[(ref_idx - 10):(ref_idx + 11)] += data[idx, :]
    centered_data /= data.shape[0]
    while (centered_data[0] == 0):
        centered_data = centered_data[1:]
    while (centered_data[-1] == 0):
        centered_data = centered_data[:-1]
    
    plt.clf()
    plt.plot(centered_data)
    if (type == "beta"):
        plt.savefig(path + mode + "/img/5/" + "centered_avg_beta" + ".png")
    else:
        plt.savefig(path + mode + "/img/5/" + "centered_avg_non_beta" + ".png")
    plt.close()

main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/5/", "beta", "beta")
main("/mnt/data/Professional/UHN/pac_investigation/results/", "/data/5/", "beta", "non beta")
print("terminated")



