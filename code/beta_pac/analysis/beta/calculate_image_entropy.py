'''
Created on May 4, 2021

@author: voodoocode
'''

import matplotlib.pyplot as plt
import numpy as np

import imageio
import scipy.ndimage

import pickle

def main():
    img1 = imageio.imread("/home/voodoocode/Downloads/img1.png")[:, :, 0]
    img2 = imageio.imread("/home/voodoocode/Downloads/img2.png")[:, :, 0]
    
#    img3 = imageio.imread("/home/voodoocode/Documents/professional/UHN/pac_investigation/results/beta/img/3/2623-s1-130.png")[:, :, 0]
#    img4 = imageio.imread("/home/voodoocode/Documents/professional/UHN/pac_investigation/results/beta/img/3/2626-s4-568-b.png")[:, :, 0]
    
    img3 = pickle.load(open("/home/voodoocode/Documents/professional/UHN/pac_investigation/results/beta/data/3/2623-s1-130.pkl", "rb"))
    img4 = pickle.load(open("/home/voodoocode/Documents/professional/UHN/pac_investigation/results/beta/data/3/2626-s4-568-b.pkl", "rb"))

    #print(calc_entropy(img1))
    #print(calc_entropy(img2))
    
    #print(calc_entropy(img3[3]))
    #print(calc_entropy(img3[4]))
    print(calc_entropy(img4[3]))
    print(calc_entropy(img4[4]))

def calc_entropy(data, h_bin_cnt = 16, v_bin_cnt = 16):
    """
    Calculates the entropy of a 2D array based on the curvature in the image.
    
    Similar to the method described in, changes:
    - Changed from a 2x2 filter matrix to a 3x3 sobel filter for edge detection
    
    Source: https://stats.stackexchange.com/questions/235270/entropy-of-an-image
    https://arxiv.org/pdf/1609.01117.pdf
    """
    data_h = scipy.ndimage.sobel(data, axis = 0)
    data_v = scipy.ndimage.sobel(data, axis = 1)
    
    data_h = data_h - np.min(data_h)
    data_h = data_h / np.max(data_h)
    
    data_v = data_v - np.min(data_v)
    data_v = data_v / np.max(data_v)
    
    if (np.isnan(data_h).any() or np.isnan(data_v).any()):
        return 0
    
    psd = np.zeros((h_bin_cnt, v_bin_cnt))
    
    for x_idx in range(0, data.shape[0]):
        for y_idx in range(0, data.shape[1]):
            x_bin = int(data_h[x_idx, y_idx]*h_bin_cnt) if (data_h[x_idx, y_idx] != np.max(data_h)) else (h_bin_cnt - 1)
            y_bin = int(data_v[x_idx, y_idx]*v_bin_cnt) if (data_v[x_idx, y_idx] != np.max(data_v)) else (v_bin_cnt - 1)
            
            psd[x_bin, y_bin] += 1
    
    psd = psd.reshape(-1)
    
    psd /= data.shape[0]*data.shape[1]
    psd = psd.transpose()
    
    return -np.sum(psd[psd.nonzero()] * np.log2(psd[psd.nonzero()]))

