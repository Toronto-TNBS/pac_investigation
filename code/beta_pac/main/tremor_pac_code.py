# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:01 2019

@author: Luka
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

import scipy.signal

import finn.filters.frequency

import pandas



#################### TREMOR CELLS ########################################################################################

######################### Vim PATIENT 1 - 671 ###############################
######################### TREMOR_NEURON 
# not bad.
# 5 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######################### TREMOR_NEURON  STIM ON... works...
# not bad.
# 3 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-on.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######################### TREMOR_NEURON  immediately after STIM... works...
# not bad.
# 4 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-off.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\671-947-stim-off.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6




######################### Vim PATIENT 2 - 669 ###############################
######################### TREMOR_NEURON 
# not bad. PUB FIG
## 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 20
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6

######################### TREMOR_NEURON STIM ON... works...
# not bad.
# 3 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\669-cell2-stim-on.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 20
#smooth_factor = 0.03
#broken_lfp = 0
#acc_channel = 6



######################### Vim PATIENT 3 - 676 ###############################
######################### TREMOR_NEURON 
# not bad.
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
#smooth_factor = 0.03
## better when not using filter (already so smooth)
#broken_lfp = 1
#acc_channel = 6

######################### TREMOR_NEURON STIM ON... works...
# not bad.
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4-stim-on.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\676-cell4-stim-on.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
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
# nice. PUB FIG
# 6 sec
hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b.txt_conv_hdr.pkl", "rb"))
data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\642-2188-b.txt_conv_data.pkl", "rb"))
filt_low = 3
filt_high = 7
smooth_factor = 0.03
broken_lfp = 0
acc_channel = 3

######################### TREMOR_NEURON STIM ON 
# works
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
# nice. pub
# 8 sec
#hdr01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\639-2376.txt_conv_hdr.pkl", "rb"))
#data01 = pickle.load(open("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\639-2376.txt_conv_data.pkl", "rb"))
#filt_low = 2
#filt_high = 8
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
#filt_low = 2
#filt_high = 8
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
#filt_low = 3
#filt_high = 6
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

#    # not so great butterworth filter
#    b,a = scipy.signal.butter(N = 2, Wn = 300/(fs_data01/2), btype = 'highpass', fs = fs_data01)
#    filtered_data01 = scipy.signal.filtfilt(b, a, data01[20])


    # using max's filter framework
    # high pass filtering raw data to isolate spikes
    hpf_data01 = finn.filters.frequency.fir(np.asarray(data01[20]), 300, None, 1, fs_data01)
    # filtering to tremor frequency band
    bpf_data01 = finn.filters.frequency.fir(np.asarray(data01[20]), filt_low, filt_high, 0.1, fs_data01)

        

    # spike signal smoothing - with padding
    winWidth = int(0.01*fs_data01*2)
    smoothed_data01 = np.pad(hpf_data01, winWidth, 'constant', constant_values = [0])
    smoothed_data01 = np.asarray(pandas.Series(np.abs(smoothed_data01)).rolling(center = True, window = int(smooth_factor*fs_data01)).mean())
    smoothed_data01 = smoothed_data01[winWidth:-winWidth]*-1
    resampled_smoothed_data01 = scipy.signal.resample(smoothed_data01, len(data01[acc_channel]), t=None, axis=0, window=None)

    
    
#    # DOUBLE SMOOTHING
#    winWidth = int(0.01*fs_data01*2)
#    smoothed_smooth = np.pad(smoothed_data01, winWidth, 'constant', constant_values = [0])
#    smoothed_smooth = np.asarray(pandas.Series(np.abs(smoothed_smooth)).rolling(center = True, window = int(smooth_factor*fs_data01)).mean())
#    smoothed_smooth = smoothed_smooth[winWidth:-winWidth]*-1



    # LPF acc, hilbert transform accelerometer
    if broken_lfp == 1:
        lpf_acc01 = data01[acc_channel]
    else:
        lpf_acc01 = finn.filters.frequency.fir(np.asarray(data01[acc_channel]), None, 6, 0.1, fs_acc01)
    hilbert_acc01 = scipy.signal.hilbert(lpf_acc01)
    inst_phase1 = np.angle(hilbert_acc01)
    
    
    # Hilbert transform the LFP and get inst phase
    hilbert_data01 = scipy.signal.hilbert(bpf_data01)
    inst_phase2 = np.angle(hilbert_data01)


    
    #FIGURE 1
    # PLOT HPF-SPK, SMOOTH, LPF
    plt.figure()
    plt.subplot(311)
    # plot spike train
    t_data01 = np.arange(1, len(hpf_data01)+1, 1)/fs_data01
    plt.plot(t_data01, hpf_data01)
    # plot resampled smoothed and accelerometer trace
    t_acc01 = np.arange(1, len(data01[acc_channel])+1, 1)/fs_acc01
    plt.plot(t_acc01, resampled_smoothed_data01)
    plt.plot(t_data01, smoothed_data01)
    #plt.plot(t_data01, smoothed_smooth)
    plt.subplot(313)
    plt.plot(t_acc01, data01[acc_channel])
    plt.plot(t_acc01, lpf_acc01)
    plt.plot(t_acc01, hilbert_acc01)
    # plot lpf
    plt.subplot(312)
    plt.plot(t_data01, bpf_data01)
    plt.show()


    #FIGURE 2
    plt.figure()
    plt.subplot(211)
    plt.plot(t_acc01, resampled_smoothed_data01*5)
    plt.plot(t_acc01, lpf_acc01)
    plt.subplot(212)
    plt.plot(t_data01, smoothed_data01)
    plt.plot(t_data01, bpf_data01)

    
    #FIGURE 3
    # PLOT PSDS (tramsform using Welch's method) and PAC-PLOT
    plt.figure()
    plt.subplot(311)
    # Welch's method of lfp to transform to frequency domain
    (freq_lpf01, bins_lpf01) = scipy.signal.welch(bpf_data01, nfft = int(fs_data01/2), fs = fs_data01, nperseg = int(fs_data01/2), noverlap = 0)
    plt.plot(freq_lpf01, np.log10(bins_lpf01))
    # Welch's method of acc to transform to frequency domain
    (freq_acc01, bins_acc01) = scipy.signal.welch(lpf_acc01, nfft = int(fs_acc01/2), fs = fs_acc01, nperseg = int(fs_acc01/2), noverlap = 0)
    plt.plot(freq_acc01, np.log10(bins_acc01))
    # Welch's method of smoothed to transform to frequency domain
    (freq_smoothed01, bins_smoothed01) = scipy.signal.welch(resampled_smoothed_data01, nfft = int(fs_acc01/2), fs = fs_acc01, nperseg = int(fs_acc01/2), noverlap = 0)
    plt.plot(freq_smoothed01, np.log10(bins_smoothed01))  
    
    
    
    ##################
    # PAC-histograms #
    ##################
    
    # generate PAC-histogram SPIKE-ACC
    winWidth = 0.2
    hist = list()         
    for x in np.arange(-np.pi, np.pi - winWidth, winWidth):
        locData = resampled_smoothed_data01[np.argwhere(np.abs(inst_phase1 - x) < winWidth).squeeze(1)]
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
    fit = scipy.optimize.curve_fit(my_sin, t_sin, hist, p0=p0)
    data_fit = my_sin(t_sin, *fit[0])
    
    # determine phase offset
    phase_offset = fit[0][1]
    print('phase offset SPIKE-ACC = ', phase_offset)
    # determine mse of PAC-hist and sine fit    
    fit_mse = np.sum(np.square(hist-data_fit))    
    print('fit mse SPIKE-ACC = ', fit_mse)    
            
    #plot PAC-histogram and sine-fit
    plt.subplot(313)
    plt.plot(hist)
    plt.plot(data_fit)
    plt.show()   
    
    
    
    # generate PAC-histogram SPIKE-LFP  
    winWidth = 0.2
    hist_lfp = list()         
    for x in np.arange(-np.pi, np.pi - winWidth, winWidth):
        locData = smoothed_data01[np.argwhere(np.abs(inst_phase2 - x) < winWidth).squeeze(1)]
        hist_lfp.append(np.mean(locData))
    # normalizing PAC-hist_lfp to have an amplitude of 1 (get rid of this to un-normalize)
    hist_lfp = hist_lfp-np.min(hist_lfp)
    hist_lfp = hist_lfp/np.max(hist_lfp)
    hist_lfp = hist_lfp*2
    hist_lfp = hist_lfp-1
    
    # fit sine to PAC-histogram
    N = len(hist_lfp)
    t_sin = np.linspace(0, 2*np.pi, N)   
    guess_amplitude = 3*np.std(hist_lfp)/(2**0.5)
    guess_phase = 0
    p0 = [guess_amplitude, guess_phase]
    def my_sin(x, amplitude, phase):
        # frequency constrained to 1
        freq = 1
        amplitude = 1
        return np.sin(x * freq + phase) * amplitude
    fit = scipy.optimize.curve_fit(my_sin, t_sin, hist_lfp, p0=p0)
    data_fit = my_sin(t_sin, *fit[0])
    
    # determine phase offset
    phase_offset = fit[0][1]
    print('phase offset SPIKE-LFP = ', phase_offset)
    # determine mse of PAC-hist_lfp and sine fit    
    fit_mse = np.sum(np.square(hist_lfp-data_fit))    
    print('fit mse SPIKE-LFP = ', fit_mse)    
            
    #plot PAC-histogram and sine-fit
    plt.subplot(312)
    plt.plot(hist_lfp)
    plt.plot(data_fit)
    plt.show()   
    
    
    
    
    
    ###################################################################
    ### SAVING DATA TO FILES ##########################################
    ###################################################################
    
    #NEEDS TO BE UPDATED
    
    # SAVING TO FILES
    # generate PSD matrix for lpf and spk then save to file  
    freq_lpf01 = freq_lpf01[:24]
    bins_lpf01 = bins_lpf01[:24]
    bins_smoothed01 = bins_smoothed01[:24]
    PSD_matrix = np.concatenate((np.expand_dims(freq_lpf01, axis = 1), np.expand_dims(bins_lpf01, axis = 1), np.expand_dims(bins_smoothed01, axis = 1)), axis = 1)
    # save PSDs to file
    np.savetxt("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\PSD_data.csv", PSD_matrix, delimiter=",")
    # generate PAC-hist and sine-fit matrix
    PAC_matrix= np.concatenate((np.expand_dims(hist, axis = 1), np.expand_dims(data_fit, axis = 1)), axis = 1)    
    # save PAC-hist to file
    np.savetxt("D:\\Desktop\\PAC_spike_lfp\\DATA\\tremor\\data_for_python\\PAC_his_data.csv", PAC_matrix, delimiter=",")
    


    print("Terminated successfully")
    
main(data01, hdr01, False)


