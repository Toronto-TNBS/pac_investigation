'''
Created on Apr 8, 2021

@author: voodoocode
'''

import pyexcel_ods
import os
import numpy as np

class ods_data():

    __data = None
    __path = None

    def __init__(self, path):
        self.__read_file(path)
        
        self.__path = path

    def __read_file(self, path):
        if (os.path.exists(path) == False):
            raise IOError("Path does not exist")        
        if (self.__data is not None or self.__path is not None):
            raise AssertionError("Object already linked to file")
        
        self.__data = pyexcel_ods.get_data(path)
        
    def get_sheet_as_dict(self, sheet):
        if (self.__data is None):
            raise AssertionError("No data loaded")
         
        labels = self.__data[sheet][0]
        ods_sheet_data = dict()
        idx = dict()
        for (label_idx, label) in enumerate(labels):
            idx[label] = label_idx
            ods_sheet_data[label] = list()
        
        for data_pt in self.__data[sheet][1:]:
            for label in labels:
                try:
                    ods_sheet_data[label].append(data_pt[idx[label]])
                except IndexError:
                    ods_sheet_data[label].append("")

    
        return ods_sheet_data
    
    def get_sheet_as_array(self, sheet):
        if (self.__data is None):
            raise AssertionError("No data loaded")
        
        labels = self.__data[sheet][0]
        data = self.__data[sheet][1:]
        
        max_length = 0
        for idx in np.arange(len(data)):
            max_length = np.max([len(data[idx]), max_length])
        
        array_data = np.zeros((len(data), max_length), dtype = np.asarray(data).dtype)
        for idx in np.arange(len(data)):
            array_data[idx, 0:len(data[idx])] = data[idx]
        
        return (labels, array_data)
    
    def modify_sheet_from_dict(self, sheet, data):
        ods_sheet_data = list()
        keys = data.keys()
        
        ods_sheet_data.append(list(data.keys()))
        for (file_idx, _) in enumerate(data[list(data.keys())[0]]):
            loc_data = list()
            for key in keys:
                loc_data.append(data[key][file_idx])
            ods_sheet_data.append(loc_data)
        
        self.__data[sheet] = ods_sheet_data
    
    def modify_sheet_from_array(self, sheet, data, labels):
        tmp = data.tolist(); tmp.insert(0, labels)
        self.__data[sheet] = tmp
        
    def write_file(self, path = None):        
        if (path is None and self.__path is not None):
            pyexcel_ods.write_data(self.__path, self.__data)
        elif (path is not None):
            pyexcel_ods.write_data(path, self.__data)
        else:
            raise AssertionError("No output path defined")



