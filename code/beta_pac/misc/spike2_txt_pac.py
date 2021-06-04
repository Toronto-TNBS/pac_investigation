'''
Created on Sep 5, 2019

@author: voodoocode
'''

import numpy as np
import pyexcel_ods
import multiprocessing
import os
import pickle

############################################################################
# conversion does not work if Evt-/+, keyboard, digimark channels present. #
############################################################################

def main(pathMetaFile, basePath, poolSz = 4, overwrite = False):
#    if (basePath[-1] != "/"):
#        basePath += "/"

    metaFile = pyexcel_ods.read_data(pathMetaFile)
    metaFile = metaFile[list(metaFile.keys())[0]]
    fileIdx = metaFile[0].index("file")

    maxRow = 0
    for rowIdx in range(1, len(metaFile)):
        if (len(metaFile[rowIdx]) != 0):
            maxRow += 1

    for rowIdx in range(1, maxRow + 1):        
        __mainInner(basePath, metaFile[rowIdx][fileIdx], overwrite)

#===============================================================================
#     pool = multiprocessing.Pool(poolSz)
#         
#     pool.starmap(__mainInner, [(basePath, metaFile[rowIdx][folderIdx], metaFile[rowIdx][fileIdx], overwrite) for rowIdx in range(1, maxRow + 1)])
# 
#     pool.close()
#     pool.join()
#===============================================================================
    

import neo
import matplotlib.pyplot as plt

def __mainInner(basePath, file, overwrite):
    filePath = basePath + file + ".smr"
    
    if (os.path.exists(filePath + "_conv_hdr.pkl") == True):
        return
    
    print(file)
    
    file = neo.Spike2IO(filePath)
    
    fs = file.get_signal_sampling_rate([0])
    raw_data = np.asarray(file.read(lazy=False)[0].segments[0].analogsignals[0])[:, 0]
    
    #plt.plot(raw_data)
    #plt.show(block = True)
    
    hdr = {20 : {"fs" : fs}}
    data = {20 : raw_data}
    
    filePath = filePath[:filePath.index(".smr")]
    pickle.dump(hdr, open(filePath + ".txt_conv_hdr.pkl", "wb"))
    pickle.dump(data, open(filePath + ".txt_conv_data.pkl", "wb"))

main("/mnt/data/Professional/UHN/pac_investigation/data/beta/new2/meta.ods",
     "/mnt/data/Professional/UHN/pac_investigation/data/beta/new2/", 4, False)
print("Terminated successfully")

















