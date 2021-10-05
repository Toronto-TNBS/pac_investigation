'''
Created on Sep 29, 2021

@author: voodoocode
'''

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats

#Positive (e.g. burst[0]) indicates the neuron 'driving'
#Negative (e.g. burst[1]) indicates the LFP 'driving'

def main(mode, threshold = 0.1):
    data = np.load("/mnt/data/Professional/UHN/pac_investigation/code/beta_pac/analysis/beta/test.npy")

    if (mode == 1 or mode == 3):
        data = data[np.asarray(data[:, 4], dtype = float) == 1, :]
    if (mode == 2 or mode == 3):
        data = data[np.asarray(data[:, 3], dtype = float) == 1, :]
    
    defaults = list()
    bursts = list()
    non_bursts = list()
    
    for x in [0, 1, 2]:
        tmp = data[np.argwhere(np.abs(np.asarray(data[:, x], dtype = float)) > threshold), x]
        tmp = np.asarray(tmp, dtype = float)
    
        if (x == 0):
            defaults.append([np.argwhere(tmp > 0).shape[0], np.argwhere(tmp < 0).shape[0]])
        if (x == 1):
            bursts.append([np.argwhere(tmp > 0).shape[0], np.argwhere(tmp < 0).shape[0]])
        if (x == 2):
            non_bursts.append([np.argwhere(tmp > 0).shape[0], np.argwhere(tmp < 0).shape[0]])
            
    defaults = np.asarray(defaults, dtype = float).squeeze()
    bursts = np.asarray(bursts, dtype = float).squeeze()
    non_bursts = np.asarray(non_bursts, dtype = float).squeeze()
    
    print("%05.2f,%05.2f,%05.2f,,%05.2f,%05.2f,%05.2f,,%05.2f,%05.2f,%05.2f" % (defaults[0], defaults[1], defaults[0]/(defaults[0]+defaults[1]), bursts[0], bursts[1], bursts[0]/(bursts[0]+bursts[1]), non_bursts[0], non_bursts[1], non_bursts[0]/(non_bursts[0]+non_bursts[1])))

    return ([defaults[0], defaults[1], defaults[0]+defaults[1], bursts[0], bursts[1], bursts[0]+bursts[1], non_bursts[0], non_bursts[1], non_bursts[0]+non_bursts[1]])

fig = plt.figure()
thresholds = [0.1, 0.125, 0.15, 0.175, 0.2]
for (mode_idx, mode) in enumerate([0, 1, 2, 3]):
    values = list()
    for (thresh_idx, threshold) in enumerate(thresholds):
        loc_vals = main(mode, threshold)
        values.append(loc_vals[3]/loc_vals[5])
        print(scipy.stats.norm.cdf(-((loc_vals[3]/loc_vals[5]) - 0.5)/(np.sqrt(0.5*(1-0.5)/loc_vals[5])))*2, loc_vals[5])
    plt.bar(x = np.arange(0, len(values)) + (len(thresholds) + 1) * mode_idx, height = values, width = 0.8)
    print("\n")
plt.show(block = True)


