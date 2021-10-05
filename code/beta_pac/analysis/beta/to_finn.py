'''
Created on Aug 20, 2021

@author: voodoocode
'''

import numpy as np
import scipy.signal

import lmfit

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
    return amp * (np.sin(2 * np.pi * freq * (x - phase / 360) / fs))

def dmi(low_freq_data, spike_data,
         phase_window_half_size = 10, 
         max_model_fit_iterations = 200):
    """
    Calculates the direct modulation index between a low frequency signal and a high frequency signal. Instead of the original modulation index based on entropy, this modulation index estimate is based on a sinusoidal fit. 
    
    :param low_freq_data: Single array of low frequency data.
    :param spike_data: Single array of spiking activity encoded as 0 and 1.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth, but also potentially increased PAC estimates.
    :param max_model_fit_iterations: Maximum number of iterations applied during sine fitting.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """

    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.arange(0, (360 - 2 * phase_window_half_size) / (phase_window_half_size * 2), 1)
    
    for (phase_bin_idx, phase_angle) in enumerate(np.arange(-180 + phase_window_half_size, 180 - phase_window_half_size, phase_window_half_size * 2)):
        phase_indices = np.argwhere(np.abs(phase_signal - phase_angle) < phase_window_half_size).squeeze(1)
        if (len(phase_indices) == 0):
            amplitude_signal[phase_bin_idx] = np.nan
        else:
            amplitude_signal[phase_bin_idx] = np.sum(spike_data[phase_indices])
                
    mod_amplitude_signal = amplitude_signal - np.nanpercentile(amplitude_signal,25)
    mod_amplitude_signal /= np.nanpercentile(mod_amplitude_signal,75)

    mod_amplitude_signal = mod_amplitude_signal * 2 - 1
    mod_amplitude_signal*= 0.70710676
    
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, min = 0.95, max = 1.05, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result = model.fit(mod_amplitude_signal, x = np.arange(0, 1, 1/len(mod_amplitude_signal)),
                       params = params, max_nfev = max_model_fit_iterations)

    if (np.isnan(mod_amplitude_signal).any() == True):
        mod_amplitude_signal = np.where(np.isnan(mod_amplitude_signal) == False)[0]

    error = np.sum(np.square(result.best_fit - mod_amplitude_signal))/len(mod_amplitude_signal)
    
    error = 1 if (error > 1) else error #Capping the error

    score = 1 - error
    score = 0 if (score < 0) else score
    
    return (score, result.best_fit, amplitude_signal)
    
    
    
    
    
    