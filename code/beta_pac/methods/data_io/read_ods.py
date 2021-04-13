'''
Created on Apr 8, 2021

@author: voodoocode
'''

import pyexcel_ods

def read_file(path, sheet):
    data = pyexcel_ods.get_data(path)[sheet]
    
    labels = data[0]
    meta_data = dict()
    idx = dict()
    for (label_idx, label) in enumerate(labels):
        idx[label] = label_idx
        meta_data[label] = list()
    
    for data_pt in data[1:]:
        for label in labels:
            try:#In case of an empty cell at the end, skip to the next element
                meta_data[label].append(data_pt[idx[label]])
            except IndexError:
                pass

    return meta_data






