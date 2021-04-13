# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: Luka
"""

import math

import numpy as np
import pickle
import matplotlib.pyplot as plt

import scipy.signal
from scipy import stats
import skimage.filters

import finn.filters.frequency

import pandas



### 3050 has nice beta
### Feb 1 935 tremor cell stn



######################### PATIENT 1 - 2626 ############################################################################### DONE
######################### BETA_NEURON 1 
# BETA nice PAC [center 21]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1 TIGHT BURSTING
#9s
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s4-568-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s4-568-b.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 16
##filt_high = 25
#smooth_factor = 0.01

######################### BETA_NEURON 2
# intermittent BETA... [center 31]
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1665-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1665-a.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### BETA_NEURON 3
# HUH-BETA [center 31] decent NOT ORIGINALLY CLASSIFIED AS BETA...
#12s
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1085.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1085.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### BETA_NEURON 4 ########### THIS IS MY TEST
# BETA... [31ish peak]
#20s
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-704.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-704.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 27
##filt_high = 37
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA... inverse
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-557.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-557.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 27
#filt_high = 37
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# NO PAC AT ALL
#hdr01 = pickle.load(open("../../data/data_for_python/2626-s2-1340.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2626-s2-1340.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 26
#filt_high = 36
#smooth_factor = 0.01



######################### BETA_NEURON 1 - LONG NEURON
# BETA nice PAC [center 21]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1 TIGHT BURSTING
#9s
hdr01 = pickle.load(open("/mnt/data/Professional/projects/PAC/data/data_for_python/2626-s4-568-long-neuron.txt_conv_hdr.pkl", "rb"))
data01 = pickle.load(open("/mnt/data/Professional/projects/PAC/data/data_for_python/2626-s4-568-long-neuron.txt_conv_data.pkl", "rb"))
data01 = data01[20]
#filt_low = 16
#filt_high = 25
smooth_factor = 0.01
# first BETA
#start = 0
#finish = 7
#data01 = data01[20][start*fs_data01:finish*fs_data01]
## no BETA
#start = 7
#finish = 13
#data01 = data01[20][start*fs_data01:finish*fs_data01]
## second BETA
#start = 14
#finish = 39
#data01 = data01[20][start*fs_data01:finish*fs_data01]





######################### PATIENT 2 - 2767 ############################################################################### DONE
######################### BETA_NEURON 1 - PUB FIG!
# nice PAC [center 24]
#hdr01 = pickle.load(open("../../data/data_for_python/2767-S1-1330-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-S1-1330-b.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 19
##filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA
#hdr01 = pickle.load(open("../../data/data_for_python/2767-s1-890-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-s1-890-a.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# NON-BETA
#hdr01 = pickle.load(open("../../data/data_for_python/2767-s1-1050-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2767-s1-1050-a.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01



######################### PATIENT 3 - 2829 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [center 14]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-1074.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-1074.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 10
##filt_high = 18
#smooth_factor = 0.01

######################### BETA_NEURON 2
# nice PAC - PUB FIGGGG!!!! [centers 16]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~5
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-1240-a.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-1240-a.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 10
#filt_high = 20
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s2-416-no-beta.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s2-416-no-beta.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 25
#filt_high = 35
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2829-s1-680-no-beta.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 10
#filt_high = 20
#smooth_factor = 0.01



######################### PATIENT 4 - 2884 ############################################################################### DONE
######################### BETA_NEURON 1...?? CONFUSING NEURON
# not bad... [peak at 16] 180 deg phase shift
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-393.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-393.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
## same neuron
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-515.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-515.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 11
#filt_high = 21
#smooth_factor = 0.01

######################### BETA_NEURON 2
# nice PAC [low center... 11?]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~6
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-615-c.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-615-c.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 8
##filt_high = 14
#smooth_factor = 0.01

######################### BETA_NEURON 3
# nice PAC [peak at 12]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~7
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s1-740.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s1-740.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 9
##filt_high = 15
#smooth_factor = 0.01

######################### BETA_NEURON 4
# dece PAC [16 center]
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-755.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-755.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 11
##filt_high = 21
#smooth_factor = 0.01

######################### BETA_NEURON 5
# dece PAC [16 center]
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-586.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-586.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 11
##filt_high = 21
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta ... no peaks
#hdr01 = pickle.load(open("../../data/data_for_python/2884-s2-945.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2884-s2-945.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 11
##filt_high = 21
#smooth_factor = 0.01


######################### PATIENT 5 - 2903 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [peak at 28] (PUB!!!!)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~8
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-650.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-650.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 24
##filt_high = 32
#smooth_factor = 0.01

######################### BETA_NEURON 2
## nice PAC [peak at 28] (PUB!!!!)
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-775-long.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-775-long.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 25
##filt_high = 34
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s2-987.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s2-987.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 25
#filt_high = 34
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# not beta [no clear peak]
#hdr01 = pickle.load(open("../../data/data_for_python/2903-s1-1078.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2903-s1-1078.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 24
#filt_high = 32
#smooth_factor = 0.01



######################### PATIENT 6 - 2623 ############################################################################### DONE
######################### BETA_NEURON 1
# nice PAC [24 peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~9
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-508.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-508.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 19
##filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 1
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-385.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-385.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 2
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s3-666.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s3-666.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 19
#filt_high = 29
#smooth_factor = 0.01

######################### NON-BETA_NEURON 3
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s4-280.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s4-280.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 21
#filt_high = 31
#smooth_factor = 0.01

######################### NON-BETA_NEURON 4
# no peak
#hdr01 = pickle.load(open("../../data/data_for_python/2623-s1-130.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2623-s1-130.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 21
#filt_high = 31
#smooth_factor = 0.01



######################### PATIENT ... - 3137 - actual ptn was 678 ###############################################################################
######################### BETA_NEURON 1
# nice PAC [24 peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~9
#hdr01 = pickle.load(open("../../data/data_for_python/3137-s-3.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/3137-s-3.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
##filt_low = 9
##filt_high = 18
#smooth_factor = 0.01

######################### PATIENT ... - 2810 ############################################################################### PRETTY CRAP
######################### BETA_NEURON 1
# fine PAC [NOT GREAT.. weak 22hz peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3
#hdr01 = pickle.load(open("../../data/data_for_python/2810-s2-950.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2810-s2-950.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 18
#filt_high = 28
#smooth_factor = 0.01
# fine PAC [NOT GREAT.. weak 22hz peak]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3
#hdr01 = pickle.load(open("../../data/data_for_python/2810-s2-950-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("../../data/data_for_python/2810-s2-950-b.txt_conv_data.pkl", "rb"))
#data01 = data01[20]
#filt_low = 8
#filt_high = 18
#smooth_factor = 0.01





def main(all_data, hdr_info, overwrite = False):

    fs_data01 = int(hdr01[20]['fs'])
    print(fs_data01)
    
    # variables for looping though time-series at one particular frequency
    # enter window size in seconds for mse calculations
    window_size = 0.5
    sample_steps = int(fs_data01 * window_size)
    
    # initiate grande mse array
    time_size = math.trunc(len(data01) / (fs_data01*window_size))
    freq_low = 13
    freq_high = 32
    freq_size = freq_high-freq_low
    filt_width = 2
    grand_mse_array = np.zeros((freq_size,time_size))    

    
    for freq_iterate in range(freq_low, freq_high):

                
        # using max's filter framework
        hpf_data01 = finn.filters.frequency.fir(np.asarray(data01), 300, None, 1, fs_data01)
        bpf_data01 = finn.filters.frequency.fir(np.asarray(data01), freq_iterate-filt_width, freq_iterate+filt_width, 0.1, fs_data01)

        
        # spike signal smoothing - with padding
        winWidth = int(0.01*fs_data01*2)
        smoothed_data01 = np.pad(hpf_data01, winWidth, 'constant', constant_values = [0]) 
        smoothed_data01 = np.asarray(pandas.Series(np.abs(smoothed_data01)).rolling(center = True, window = int(smooth_factor*fs_data01)).mean())
        smoothed_data01 = smoothed_data01[winWidth:-winWidth]*-1

        
        # PLOT HPF-SPK, SMOOTH, LPF
    #    plt.subplot(211)
    #    t_data01 = np.arange(1, len(hpf_data01)+1, 1)/fs_data01
    #    plt.plot(t_data01, hpf_data01)
    #    plt.plot(t_data01, smoothed_data01)
    #    plt.subplot(212)
    #    plt.plot(t_data01, bpf_data01)
    #    plt.show()  
    
        # PLOT PSD of Smooth signal and LFP (tramsform using Welch's method) and PAC-PLOT
    #    plt.figure()
    #    # PSD
    #    # plt.subplot(211)
    #    # Welch's method of transform to frequency domain
    #    (freq_lpf01, bins_lpf01) = scipy.signal.welch(bpf_data01, nfft = int(fs_data01/2), fs = fs_data01, nperseg = int(fs_data01/2), noverlap = 0)
    #    plt.plot(freq_lpf01, np.log10(bins_lpf01))
    #    (freq_smoothed01, bins_smoothed01) = scipy.signal.welch(smoothed_data01, nfft = int(fs_data01/2), fs = fs_data01, nperseg = int(fs_data01/2), noverlap = 0)
    #    plt.plot(freq_smoothed01, np.log10(bins_smoothed01))  

        
        time_series_mse = []  
        time_iteration_factor = 0                

        for time_iterate in range(0, int(len(bpf_data01)/sample_steps)):    
                
            smoothed_data01_loc = smoothed_data01[0 + (sample_steps*time_iteration_factor) : sample_steps + (sample_steps*time_iteration_factor)]
            
            # Hilbert transform LFP and get inst phase
            bpf_data01_loc = bpf_data01[0 + (sample_steps*time_iteration_factor) : sample_steps + (sample_steps*time_iteration_factor)]
            hilbert_data01 = scipy.signal.hilbert(bpf_data01_loc)
            inst_phase = np.angle(hilbert_data01)
            
          
            # generate PAC-histogram    
            winWidth = 0.2
            hist = list()    
            for x in np.arange(-np.pi, np.pi - winWidth, winWidth):
                locData = smoothed_data01_loc[np.argwhere(np.abs(inst_phase - x) < winWidth).squeeze(1)]
                hist.append(np.mean(locData))
            # normalizing PAC-hist to have an amplitude of 1 (get rid of this to un-normalize)
            hist = hist-np.min(hist)
            hist = hist/np.max(hist)
            hist = hist*2
            hist = hist-1
                
            
            # fit sine to PAC-histogram
            N = len(hist)
            t_sin = np.linspace(0, 2*np.pi, N)   
            guess_amplitude = 3*np.std(hist)/(2**0.5)
            guess_phase = 0
            p0 = [guess_amplitude, guess_phase]
        
            def my_sin(x, amplitude, phase):
                # frequency constrained to 1
                freq = 1
                amplitude = 1
                return np.sin(x * freq + phase) * amplitude
            
            # if parameter optimiziation fails, manually set MSE to 20.                        
            try:
                fit = scipy.optimize.curve_fit(my_sin, t_sin, hist, p0=p0)
                data_fit = my_sin(t_sin, *fit[0])
                ##determine phase offset
                #phase_offset = fit[0][1]
                #print('phase offset = ', phase_offset)
                ##determine mse of PAC-hist and sine fit    
                fit_mse = np.sum(np.square(hist-data_fit))    
                #print('fit mse = ', fit_mse)    
            except RuntimeError:
                fit_mse = 20          
            

                                
            time_series_mse.append(fit_mse)
            time_iteration_factor += 1               


            
        time_series_mse = np.array(time_series_mse)
        grand_mse_array[freq_iterate-freq_low,:] = time_series_mse
    
    
    
    def heatmap2d(arr: np.ndarray):
        plt.figure()
        plt.imshow(arr, interpolation = 'gaussian', cmap='afmhot', aspect = 'auto', vmax = 5)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()
    
    heatmap2d(grand_mse_array)
    
    
    
    ###################################################################
    ### SAVING DATA TO FILES ##########################################
    ###################################################################
    
#    # SAVING TO FILES
#    # generate PSD matrix for lpf and spk then save to file  
#    freq_lpf01 = freq_lpf01[:24]
#    bins_lpf01 = bins_lpf01[:24]
#    bins_smoothed01 = bins_smoothed01[:24]
#    PSD_matrix = np.concatenate((np.expand_dims(freq_lpf01, axis = 1), np.expand_dims(bins_lpf01, axis = 1), np.expand_dims(bins_smoothed01, axis = 1)), axis = 1)
#    # save PSDs to file
#    np.savetxt("../../data/data_for_python/PSD_data.csv", PSD_matrix, delimiter=",")
#    # generate PAC-hist and sine-fit matrix
#    PAC_matrix= np.concatenate((np.expand_dims(hist, axis = 1), np.expand_dims(data_fit, axis = 1)), axis = 1)    
#    # save PAC-hist to file
#    np.savetxt("../../data/data_for_python/PAC_his_data.csv", PAC_matrix, delimiter=",")
    


    print("Terminated successfully")
    
main(data01, hdr01, False)


