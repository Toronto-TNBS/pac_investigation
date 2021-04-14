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
    folderIdx = metaFile[0].index("folder")
    fileIdx = metaFile[0].index("file")

    maxRow = 0
    for rowIdx in range(1, len(metaFile)):
        if (len(metaFile[rowIdx]) != 0):
            maxRow += 1

    for rowIdx in range(1, maxRow + 1):
        __mainInner(basePath, metaFile[rowIdx][folderIdx], metaFile[rowIdx][fileIdx], overwrite)

#===============================================================================
#     pool = multiprocessing.Pool(poolSz)
#         
#     pool.starmap(__mainInner, [(basePath, metaFile[rowIdx][folderIdx], metaFile[rowIdx][fileIdx], overwrite) for rowIdx in range(1, maxRow + 1)])
# 
#     pool.close()
#     pool.join()
#===============================================================================
    
    print("Terminated successfully")

def __mainInner(basePath, folder, file, overwrite):
#    if (folder[-1] != "/"):
#        folder += "/"

    filePath = basePath + folder + "\\" + file + ".txt"
    
    if (os.path.exists(filePath + "_conv_data.pkl") and os.path.exists(filePath + "_conv_hdr.pkl") and overwrite == False):
        return
    
    print("Processing file %s" % (filePath, ))
    (hdr, data) = parseFile(filePath)
    
    pickle.dump(hdr, open(filePath + "_conv_hdr.pkl", "wb"))
    pickle.dump(data, open(filePath + "_conv_data.pkl", "wb"))
    
def parseFile(filePath):
    file = open(filePath, "r")
    
    while (True):
        line = file.readline()
        if (line == '"SUMMARY"\n'):
            break
        
    hdr     = dict()
    data    = dict()
        
    while (True):
        line = file.readline()
        if (line == "\n"):
            break

        line = line.split("\t")
        
        idx = int(line[0].replace("\"", "").replace("\n",""))
        sigType = line[1].replace("\"", "").replace("\n","")
        name = line[2].replace("\"", "").replace("\n","")
        unknownAttribute1 = line[3].replace("\"", "").replace("\n","")
        fs = float(line[5].replace("\"", "").replace("\n",""))
        unknownAttribute2 = line[6].replace("\"", "").replace("\n","")
        unknownAttribute3 = line[7].replace("\"", "").replace("\n","")
        
        hdr[idx]  = {"type" : sigType, "name" : name, "unknownAttribute1" : unknownAttribute1, "fs" : fs, "unknownAttribute2" : unknownAttribute2, "unknownAttribute3" : unknownAttribute3}
        data[idx] = list()
    
    while(True):
        while(True):
            line = file.readline()
            if ("CHANNEL" in line):
                break
            
            if(len(line) == 0):
                file.close()
                return (hdr, data)

        idx = int(line.split("\t")[1].replace("\"", "").replace("\n",""))
                
        while(True):
            line = file.readline()
            if ("START" in line):
                break
        
        while(True):
            line = file.readline()
            
            if (line == "\n"):
                break
            
            data[idx].append(float(line.replace("\"", "").replace("\n","")))



main("D:\\Users\\lukam\\Desktop\\PAC_spike_lfp\\DATA\\data_for_python\\meta_data_PAC_test.ods", "D:\\Users\\lukam\\Desktop\\PAC_spike_lfp\\DATA\\", 4, False)

















