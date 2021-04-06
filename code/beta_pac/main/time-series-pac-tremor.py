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

from random import randint


#################### TREMOR CELLS ########################################################################################

######################### Vim PATIENT 1 - 671 ###############################
######################### TREMOR_NEURON 
# not bad.
# 5 sec
hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947.txt_conv_hdr.pkl", "rb"))
data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
smooth_factor = 0.03
broken_lfp = 0
acc_channel = 6

######################### TREMOR_NEURON  STIM ON... works...
# not bad.
# 3 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-on.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######################### TREMOR_NEURON  immediately after STIM... works...
# not bad.
# 4 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-off.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-off.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6




######################### Vim PATIENT 2 - 669 ###############################
######################### TREMOR_NEURON 
# not bad.
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 20
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######################### TREMOR_NEURON STIM ON... works...
# not bad.
# 3 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2-stim-on.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 20
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6



######################### Vim PATIENT 3 - 676 ###############################
######################### TREMOR_NEURON 
# not bad.
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
## better when not using filter (already so smooth)
#broken_lfp = 1
#acc_channel = 6

######################### TREMOR_NEURON STIM ON... works...
# not bad.
# 5 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4-stim-on.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
## better when not using filter (already so smooth)
#broken_lfp = 1
#acc_channel = 6




######################### Vim PATIENT 4 - 664 ###############################
######################### TREMOR_NEURON 
# beautiful with cardiac fixed
# 6 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\664-cell1-c.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\664-cell1-c.txt_conv_data.pkl", "rb"))
#filt_low = 3
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 1
#acc_channel = 6
## fix cardio-ballistic artifact
#for x in range(0, len(data01[20])-1):
#    if data01[20][x] <= -0.1 and data01[20][x] >= -0.2:
#        data01[20][x] = -0.4

######################### TREMOR_NEURON STIM ON... works...
# works
# 6 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\664-cell1-c-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\664-cell1-c-stim-on.txt_conv_data.pkl", "rb"))
#filt_low = 3
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 1
#acc_channel = 6



######################### Vim PATIENT 5 - 642 ###############################
######################### TREMOR_NEURON 
# ACC not bad. LFP pretty bad.
# 6 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b.txt_conv_data.pkl", "rb"))
##filt_low = 3
##filt_high = 7
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 3

######################### TREMOR_NEURON STIM ON 
# ACC not bad. LFP pretty bad.
# 2 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b-stim-on.txt_conv_data.pkl", "rb"))
#filt_low = 3
#filt_high = 6
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6



######################### Vim PATIENT 6 - 639 ###############################
######################### TREMOR_NEURON 
# ACC not bad. LFP pretty bad.
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\639-2376.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\639-2376.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 3

# OLD DATA NO STIM ON



######################### Vim PATIENT 7 - 633 ###############################
######################### TREMOR_NEURON 
# nice
# 13 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\633-s1-1193.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\633-s1-1193.txt_conv_data.pkl", "rb"))
##filt_low = 2
##filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 3

# OLD DATA NO STIM ON


######################### Vim PATIENT 8 - 632 ###############################
######################### TREMOR_NEURON 
# ACC GOOD. LFP NOT
# 12 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\632-2085.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\632-2085.txt_conv_data.pkl", "rb"))
##filt_low = 3
##filt_high = 6
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 3

# OLD DATA NO STIM ON


#################### NON-TREMOR CELLS ########################################################################################




####################################
########## NOT SO GOOD CELLS/TREMORS

# not bad but acc twice as fast as tremor
######################### Vim PATIENT 1 - 668 ###############################
######################### TREMOR_NEURON 
# not bad but acc twice as fast as tremor
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\668-650.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\668-650.txt_conv_data.pkl", "rb"))
#filt_low = 3
#filt_high = 8
#smooth_factor = 0.04
#broken_lfp = 1
#acc_channel = 6


# HUGE ARTIFACTS NEED TO BE REMOVED (tremor kind of sucky)
######################### Vim PATIENT 6 - 635 ###############################
######################### TREMOR_NEURON 
# ACC not bad. LFP pretty bad.
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\635-s1-2831.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\635-s1-2831.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 3

######### SHITTY TREMOR TRACE... DOESN'T WORK GREAT
######################### Vim PATIENT 5 - 676 ###############################
######################### TREMOR_NEURON 
# bad.
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\647-689.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\647-689.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.02
#broken_lfp = 1
#acc_channel = 6
######################### TREMOR_NEURON 
# bad.
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\647-1076.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\647-1076.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######### FILE TOO SHORT WITH NOT GOOD TREMOR
######################### Vim PATIENT - 678 ###############################
######################### TREMOR_NEURON 
# bad.
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\678-cell1-b.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\678-cell1-b.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.04
#broken_lfp = 0
#acc_channel = 6

######################### STN PATIENT 1 - 2965 ###############################
######################### TREMOR_NEURON
# very nice [STN]
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\2965-s1-215.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\2965-s1-215.txt_conv_data.pkl", "rb"))
#filt_low = 3
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6





def main(all_data, hdr_info, overwrite = False):

    fs_data01 = int(hdr01[20]['fs'])
    print(fs_data01)
    
    fs_acc01 = int(hdr01[acc_channel]['fs'])
    print(fs_acc01)
    
    
    # variables for looping though time-series at one particular frequency
    # enter window size in seconds for mse calculations
    window_size = 0.5
    sample_steps_lfp = int(fs_data01 * window_size)
    sample_steps_tremor = int(fs_acc01 * window_size)
    
    # initiate grande mse array
    time_size = math.trunc(len(data01[20]) / (fs_data01*window_size))
    freq_low = 2
    freq_high = 10
    freq_size = freq_high-freq_low
    filt_width = 2
    grand_mse_array_lfp = np.zeros((freq_size,time_size))    
    grand_mse_array_tremor = np.zeros((freq_size,time_size)) 
    
    
        
    for freq_iterate in range(freq_low, freq_high):

                
        # using max's filter framework
        hpf_data01 = finn.filters.frequency.fir(np.asarray(data01[20]), 300, None, 1, fs_data01)
        bpf_data01 = finn.filters.frequency.fir(np.asarray(data01[20]), freq_iterate-filt_width, freq_iterate+filt_width, 0.1, fs_data01)

        
        # LPF acc, hilbert transform accelerometer
        if broken_lfp == 1:
            lpf_acc01 = data01[acc_channel]
        else:
            lpf_acc01 = finn.filters.frequency.fir(np.asarray(data01[acc_channel]), freq_iterate-filt_width, freq_iterate+filt_width, 0.1, fs_acc01)

        
        
        # spike signal smoothing - with padding
        winWidth = int(0.01*fs_data01*2)
        smoothed_data01 = np.pad(hpf_data01, winWidth, 'constant', constant_values = [0]) 
        smoothed_data01 = np.asarray(pandas.Series(np.abs(smoothed_data01)).rolling(center = True, window = int(smooth_factor*fs_data01)).mean())
        smoothed_data01 = smoothed_data01[winWidth:-winWidth]*-1
        resampled_smoothed_data01 = scipy.signal.resample(smoothed_data01, len(data01[acc_channel]), t=None, axis=0, window=None)


        time_series_mse_lfp = []
        time_series_mse_tremor = []
        time_iteration_factor_lfp = 0
        time_iteration_factor_tremor = 0                


        for time_iterate_lfp in range(0, int(len(bpf_data01)/sample_steps_lfp)):    

            smoothed_data01_loc = smoothed_data01[0 + (sample_steps_lfp*time_iteration_factor_lfp) : sample_steps_lfp + (sample_steps_lfp*time_iteration_factor_lfp)]
        
            # Hilbert transform the LFP and get inst phase
            bpf_data01_loc = bpf_data01[0 + (sample_steps_lfp*time_iteration_factor_lfp) : sample_steps_lfp + (sample_steps_lfp*time_iteration_factor_lfp)]
            hilbert_data01 = scipy.signal.hilbert(bpf_data01_loc)
            inst_phase2 = np.angle(hilbert_data01)
            
            
            # generate PAC-histogram SPIKE-LFP  
            winWidth = 0.2
            hist_lfp = list()         
            for x1 in np.arange(-np.pi, np.pi - winWidth, winWidth):
                locData = smoothed_data01_loc[np.argwhere(np.abs(inst_phase2 - x1) < winWidth).squeeze(1)]
                #if (np.isnan(locData).any() == True):
                    #hist_lfp.append(error_val)
                #else:
                hist_lfp.append(np.mean(locData))
            ### BUG FIX
            if (np.isnan(hist_lfp).any() == True):
                hist_lfp = list()
                for y in np.arange(-np.pi, np.pi - winWidth, winWidth):
                    error_val = randint(0, 10)
                    hist_lfp.append(error_val)
                    
                    
            # normalizing PAC-hist_lfp to have an amplitude of 1 (get rid of this to un-normalize)
            hist_lfp = hist_lfp-np.min(hist_lfp)
            hist_lfp = hist_lfp/np.max(hist_lfp)
            hist_lfp = hist_lfp*2
            hist_lfp = hist_lfp-1
            
            # fit sine to PAC-histogram
            N_lfp = len(hist_lfp)
            t_sin_lfp = np.linspace(0, 2*np.pi, N_lfp)   
            #guess_amplitude_lfp = 3*np.std(hist_lfp)/(2**0.5)
            guess_amplitude_lfp = 1
            guess_phase_lfp = 0
            p0_lfp = [guess_amplitude_lfp, guess_phase_lfp]
            
            def my_sin_lfp(x, amplitude, phase):
                # frequency and amplitude constrained to 1
                freq = 1
                amplitude = 1
                return np.sin(x * freq + phase) * amplitude
                        
            # if parameter optimiziation fails, manually set MSE to 20.                        
            try:
                fit_lfp = scipy.optimize.curve_fit(my_sin_lfp, t_sin_lfp, hist_lfp, p0=p0_lfp)
                data_fit_lfp = my_sin_lfp(t_sin_lfp, *fit_lfp[0])   
                fit_mse_lfp = np.sum(np.square(hist_lfp-data_fit_lfp))      
            except RuntimeError:
                fit_mse_lfp = 20
            
    
            time_series_mse_lfp.append(fit_mse_lfp)
            
            time_iteration_factor_lfp += 1        
    
        
        time_series_mse_lfp = np.array(time_series_mse_lfp)
        grand_mse_array_lfp[freq_iterate-freq_low,:] = time_series_mse_lfp
        
        

        for time_iterate_tremor in range(0, int(len(lpf_acc01)/sample_steps_tremor)):    
   
            resampled_smoothed_data01_loc = resampled_smoothed_data01[0 + (sample_steps_tremor*time_iteration_factor_tremor) : sample_steps_tremor + (sample_steps_tremor*time_iteration_factor_tremor)]

            ## Hilbert transform the ACC and get inst phase
            lpf_acc01_loc = lpf_acc01[0 + (sample_steps_tremor*time_iteration_factor_tremor) : sample_steps_tremor + (sample_steps_tremor*time_iteration_factor_tremor)]
            hilbert_acc01 = scipy.signal.hilbert(lpf_acc01_loc)
            inst_phase1 = np.angle(hilbert_acc01)
            
            
            # generate PAC-histogram SPIKE-LFP  
            hist_tremor = list()         
            for x2 in np.arange(-np.pi, np.pi - winWidth, winWidth):
                locData = resampled_smoothed_data01_loc[np.argwhere(np.abs(inst_phase1 - x2) < winWidth).squeeze(1)]
                hist_tremor.append(np.mean(locData))
            ### BUG FIX
            if (np.isnan(hist_tremor).any() == True):
                hist_tremor = list()
                for z in np.arange(-np.pi, np.pi - winWidth, winWidth):
                    error_val2 = randint(0, 10)
                    hist_tremor.append(error_val2)
            
            
            # normalizing PAC-hist_lfp to have an amplitude of 1 (get rid of this to un-normalize)
            hist_tremor = hist_tremor-np.min(hist_tremor)
            hist_tremor = hist_tremor/np.max(hist_tremor)
            hist_tremor = hist_tremor*2
            hist_tremor = hist_tremor-1
            
            # fit sine to PAC-histogram
            N_tremor = len(hist_tremor)
            t_sin_tremor = np.linspace(0, 2*np.pi, N_tremor)   
            guess_amplitude_tremor = 3*np.std(hist_tremor)/(2**0.5)
            guess_phase_tremor = 0
            p0_tremor = [guess_amplitude_tremor, guess_phase_tremor]
            
            def my_sin_tremor(x, amplitude, phase):
                # frequency constrained to 1
                freq = 1
                amplitude = 1
                return np.sin(x * freq + phase) * amplitude
                        
            # if parameter optimiziation fails, manually set MSE to 20.                        
            try:
                fit_tremor = scipy.optimize.curve_fit(my_sin_tremor, t_sin_tremor, hist_tremor, p0=p0_tremor)
                data_fit_tremor = my_sin_tremor(t_sin_tremor, *fit_tremor[0])   
                fit_mse_tremor = np.sum(np.square(hist_tremor-data_fit_tremor))      
            except RuntimeError:
                fit_mse_tremor = 20          
            

    
            time_series_mse_tremor.append(fit_mse_tremor)
            
            time_iteration_factor_tremor += 1        
                  
            
        time_series_mse_tremor = np.array(time_series_mse_tremor)
        grand_mse_array_tremor[freq_iterate-freq_low,:] = time_series_mse_tremor       
        
    

    #PLOTTING
    
    plt.figure()
    
    plt.subplot(211)
    plt.imshow(grand_mse_array_lfp, interpolation = 'gaussian', cmap='afmhot', aspect = 'auto', vmax = 10)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    
    plt.subplot(212)
    plt.imshow(grand_mse_array_tremor, interpolation = 'gaussian', cmap='afmhot', aspect = 'auto', vmax = 10)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    
    
    
    
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
#    np.savetxt("D:\\Desktop\\PAC_spike_lfp\\DATA\\data_for_python\\PSD_data.csv", PSD_matrix, delimiter=",")
#    # generate PAC-hist and sine-fit matrix
#    PAC_matrix= np.concatenate((np.expand_dims(hist, axis = 1), np.expand_dims(data_fit, axis = 1)), axis = 1)    
#    # save PAC-hist to file
#    np.savetxt("D:\\Desktop\\PAC_spike_lfp\\DATA\\data_for_python\\PAC_his_data.csv", PAC_matrix, delimiter=",")
    


    print("Terminated successfully")
    
main(data01, hdr01, False)