'''
Created on Dec 17, 2021

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.signal

freq = 20
fs = 1000
phase = [-110, 70]
x = np.arange(0, 10000)

(fig, axes0) = plt.subplots(5, 1)
(fig, axes1) = plt.subplots(1, 1)

colors = ["purple", "blue", "green", "orange", "red"]

noise1 = np.random.random(len(x))
noise2 = np.random.random(len(x))
noise_amp = 0.75

values = list()

for (amp_idx, amp) in enumerate([[1, 0], [1, 1/2], [1, 3/4], [1, 7/8], [1, 1]]):
    beta1 = amp[0] * (np.sin(2 * np.pi * freq * (x - (phase[0]/360)*(fs/freq)) / fs))
    beta2 = amp[1] * (np.sin(2 * np.pi * freq * (x - (phase[1]/360)*(fs/freq)) / fs))
    
    signal = np.average(np.asarray([beta1, beta2, noise1 * noise_amp/2, noise2 * noise_amp/2]), axis = 0)
    
    axes0[amp_idx].plot(np.average(np.asarray([beta1, noise1 * noise_amp]), axis = 0))
    axes0[amp_idx].plot(np.average(np.asarray([beta2, noise2 * noise_amp]), axis = 0))
    axes0[amp_idx].plot(signal)
    axes0[amp_idx].set_xlim(0, 100)
    
    (bins, power) = scipy.signal.welch(signal, fs, "hanning", int(fs), fs//2, int(fs))
    axes1.plot(bins[:50], np.log(power[:50]))
    values.append(power[20])
    
    plt.psd(signal, fs, fs)
values = np.asarray(values)
print(1-values[1:]/values[0])
plt.show(block = True)


