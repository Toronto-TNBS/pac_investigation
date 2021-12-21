'''
Created on May 26, 2021

@author: voodoocode
'''

import numpy as np
import methods.data_io.ods as ods_reader

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import sklearn.naive_bayes

def main():
    
    (meta_file, meta_labels, meta_data, lfp_data, hf_default_data, hf_burst_data, hf_non_burst_data, txt_data) = get_data()
    
    auto_classify(lfp_data)
    auto_classify(hf_default_data)
    auto_classify(hf_burst_data)
    auto_classify(hf_non_burst_data)
    
    lf_auto_idx = meta_labels.index("lf auto")
    hf_auto_idx = meta_labels.index("hf auto")
    burst_auto_idx = meta_labels.index("burst auto")
    non_burst_auto_idx = meta_labels.index("non burst auto")
    
    for (row_idx, _) in enumerate(lfp_data[:, 0]):
        meta_row_idx = np.argwhere(meta_data[:, 0] == txt_data[row_idx, 0])
        meta_data[meta_row_idx, lf_auto_idx] = lfp_data[row_idx, 1]
        meta_data[meta_row_idx, hf_auto_idx] = hf_default_data[row_idx, 1]
        meta_data[meta_row_idx, burst_auto_idx] = hf_burst_data[row_idx, 1]
        meta_data[meta_row_idx, non_burst_auto_idx] = hf_non_burst_data[row_idx, 1]
    
    #--------- meta_file.modify_sheet_from_array("beta", meta_data, meta_labels)
    #---------------------------------------------------- meta_file.write_file()
    
    visualization(lfp_data, hf_default_data, txt_data)

def auto_classify(data):
    gnb = sklearn.naive_bayes.GaussianNB()
    gnb.fit(X = np.asarray(data[np.argwhere(np.asarray(data[:, 1], dtype = np.int) != -1), 0], dtype = np.float), y = np.asarray(data[np.argwhere(np.asarray(data[:, 1], dtype = np.int) != -1), 1], dtype = np.int).ravel())    
    data[np.argwhere(np.asarray(data[:, 1], dtype = np.int) == -1), 1] = np.expand_dims(gnb.predict(data[np.argwhere(np.asarray(data[:, 1], dtype = np.int) == -1), 0]), axis = 1)
    
    return data
    
def get_data(path = "../../../../data/meta.ods", sheet = "beta"):
    full_data = ods_reader.ods_data(path)
    (pre_labels, pre_data) = full_data.get_sheet_as_array(sheet)
    
    valid_idx = pre_labels.index("valid_data")
    file_idx = pre_labels.index("file")
    
    lfp_beta_idx = pre_labels.index("beta lfp strength 1")
    hf_default_beta_idx = pre_labels.index("beta overall strength 1")
    hf_burst_beta_idx = pre_labels.index("beta burst strength 1")
    hf_non_burst_beta_idx = pre_labels.index("beta non burst strength 1")
    
    lf_3_index = pre_labels.index("lf manual")
    hf_3_index = pre_labels.index("hf manual")
    
    lfp_data = np.asarray(pre_data[np.argwhere(np.asarray(pre_data[:, valid_idx], dtype = np.int) == 1), [lfp_beta_idx, lf_3_index]], dtype = np.float32)
    hf_default_data = np.asarray(pre_data[np.argwhere(np.asarray(pre_data[:, valid_idx], dtype = np.int) == 1), [hf_default_beta_idx, hf_3_index]], dtype = np.float32)
    hf_burst_data = np.asarray(pre_data[np.argwhere(np.asarray(pre_data[:, valid_idx], dtype = np.int) == 1), [hf_burst_beta_idx, hf_3_index]], dtype = np.float32)
    hf_non_burst_data = np.asarray(pre_data[np.argwhere(np.asarray(pre_data[:, valid_idx], dtype = np.int) == 1), [hf_non_burst_beta_idx, hf_3_index]], dtype = np.float32)
    txt_data = pre_data[np.argwhere(np.asarray(pre_data[:, valid_idx], dtype = np.int) == 1), [file_idx, hf_3_index]]
    
    return (full_data, pre_labels, pre_data, lfp_data, hf_default_data, hf_burst_data, hf_non_burst_data, txt_data)
    
def visualization(lfp_data, hf_default_data, txt_data, annotate = False):
    tmp = lfp_data
    plt.figure()
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == 0), 0], tmp[np.argwhere(tmp[:, 1] == 0), 0], color = "red")
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == 1), 0], tmp[np.argwhere(tmp[:, 1] == 1), 0], color = "blue")
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == -1), 0], tmp[np.argwhere(tmp[:, 1] == -1), 0]-.5, color = "green")
    if (annotate):
        for i, txt in enumerate(txt_data[:, 0]):
            plt.annotate(txt, (tmp[:, 0][i], tmp[:, 0][i]))
    plt.title("lf")

    tmp = hf_default_data
    plt.figure()
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == 0), 0], tmp[np.argwhere(tmp[:, 1] == 0), 0], color = "red")
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == 1), 0], tmp[np.argwhere(tmp[:, 1] == 1), 0], color = "blue")
    plt.scatter(tmp[np.argwhere(tmp[:, 1] == -1), 0], tmp[np.argwhere(tmp[:, 1] == -1), 0]-.5, color = "green")
    if (annotate):
        for i, txt in enumerate(txt_data[:, 0]):
            plt.annotate(txt, (tmp[:, 0][i], tmp[:, 0][i]))
    plt.title("hf")

    
    plt.show(block = True)    
main()