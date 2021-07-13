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
    acc_ch_idx = metaFile[0].index("acc channel")

    maxRow = 0
    for rowIdx in range(1, len(metaFile)):
        if (len(metaFile[rowIdx]) != 0):
            maxRow += 1

    for rowIdx in range(1, maxRow + 1):        
        __mainInner(basePath, metaFile[rowIdx][fileIdx], metaFile[rowIdx][acc_ch_idx], overwrite)

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

magic_numbers = {
    "scale_multiplier" : 6553.6
    }

def read_file_info(file):
    file_info = dict()
    
    file_info["system_id"]          = file[(  0):(  2)].view(dtype = "<i2")[0]
    file_info["copyright"]          = file[(  2):( 12)].view(dtype = "S10")[0]
    file_info["creator"]            = file[( 12):( 20)].view(dtype =  "S8")[0]
    file_info["us_per_time"]        = file[( 20):( 22)].view(dtype = "<i2")[0]
    file_info["time_per_adc"]       = file[( 22):( 24)].view(dtype = "<i2")[0]
    file_info["filestate"]          = file[( 24):( 26)].view(dtype = "<i2")[0]
    file_info["first_data"]         = file[( 26):( 30)].view(dtype = "<i4")[0]
    file_info["channels"]           = file[( 30):( 32)].view(dtype = "<i2")[0]
    file_info["chan_size"]          = file[( 32):( 34)].view(dtype = "<i2")[0]
    file_info["extra_data"]         = file[( 34):( 36)].view(dtype = "<i2")[0]
    file_info["buffersize"]         = file[( 36):( 38)].view(dtype = "<i2")[0]
    file_info["os_format"]          = file[( 38):( 40)].view(dtype = "<i2")[0]
    file_info["max_ftime"]          = file[( 40):( 44)].view(dtype = "<i4")[0]
    file_info["dtime_base"]         = file[( 44):( 52)].view(dtype = "<f8")[0]
    file_info["datetime_detail"]    = file[( 52):( 53)].view(dtype = "<u1")[0]
    file_info["datetime_year"]      = file[( 53):( 55)].view(dtype = "<i2")[0]
    file_info["pad"]                = file[( 55):(107)].view(dtype = "S52")[0]
    file_info["comment1"]           = file[(107):(187)].view(dtype = "S80")[0]
    file_info["comment2"]           = file[(187):(267)].view(dtype = "S80")[0]
    file_info["comment3"]           = file[(267):(347)].view(dtype = "S80")[0]
    file_info["comment4"]           = file[(347):(427)].view(dtype = "S80")[0]
    file_info["comment5"]           = file[(427):(507)].view(dtype = "S80")[0]
    
    
    #===========================================================================
    # file_info["chan_size"] = np.frombuffer(file.read(2), dtype = "<i2")[0]
    # file_info["extra_data"] = np.frombuffer(file.read(2), dtype = "<i2")[0]
    # file_info["buffersize"] = np.frombuffer(file.read(2), dtype = "<i2")[0]
    # file_info["os_format"] = np.frombuffer(file.read(2), dtype = "<i2")[0]
    # file_info["max_ftime"] = np.frombuffer(file.read(4), dtype = "<i4")[0]
    # file_info["dtime_base"] = np.frombuffer(file.read(8), dtype = "f8")[0]
    # file_info["datetime_detail"] = np.frombuffer(file.read(1), dtype = "u1")[0]
    # file_info["datetime_year"] = np.frombuffer(file.read(2), dtype = "<i2")[0]
    # file_info["pad"] = str(file.read(52), encoding = "iso-8859-1")
    # file_info["comment1"] = str(file.read(80), encoding = "iso-8859-1")
    # file_info["comment2"] = str(file.read(80), encoding = "iso-8859-1")
    # file_info["comment3"] = str(file.read(80), encoding = "iso-8859-1")
    # file_info["comment4"] = str(file.read(80), encoding = "iso-8859-1")
    # file_info["comment5"] = str(file.read(80), encoding = "iso-8859-1")
    #===========================================================================
    
    return file_info

def read_channel_header(file, file_info):
    
    ch_infos = list()
    for ch_idx in range(file_info["channels"]):
        loc_start = 512 + 140 * ch_idx
        
        ch_info = dict()
        ch_info["del_size"]         = file[(loc_start +   0):(loc_start +   2)].view(dtype = "<i2")[0]
        ch_info["next_del_block"]   = file[(loc_start +   2):(loc_start +   6)].view(dtype = "<i4")[0]
        ch_info["firstblock"]       = file[(loc_start +   6):(loc_start +  10)].view(dtype = "<i4")[0]
        ch_info["lastblock"]        = file[(loc_start +  10):(loc_start +  14)].view(dtype = "<i4")[0]
        ch_info["blocks"]           = file[(loc_start +  14):(loc_start +  16)].view(dtype = "<i2")[0]
        ch_info["n_extra"]          = file[(loc_start +  16):(loc_start +  18)].view(dtype = "<i2")[0]
        ch_info["pre_trig"]         = file[(loc_start +  18):(loc_start +  20)].view(dtype = "<i2")[0]
        ch_info["free0"]            = file[(loc_start +  20):(loc_start +  22)].view(dtype = "<i2")[0]
        ch_info["py_sz"]            = file[(loc_start +  22):(loc_start +  24)].view(dtype = "<i2")[0]
        ch_info["max_data"]         = file[(loc_start +  24):(loc_start +  26)].view(dtype = "<i2")[0]
        ch_info["comment"]          = file[(loc_start +  26):(loc_start +  98)].view(dtype = "S72")[0]
        ch_info["max_chan_time"]    = file[(loc_start +  98):(loc_start + 102)].view(dtype = "<i4")[0]
        ch_info["l_chan_dvd"]       = file[(loc_start + 102):(loc_start + 106)].view(dtype = "<i4")[0]
        ch_info["phy_chan"]         = file[(loc_start + 106):(loc_start + 108)].view(dtype = "<i2")[0]
        ch_info["title"]            = file[(loc_start + 108):(loc_start + 118)].view(dtype = "S10")[0]
        ch_info["ideal_rate"]       = file[(loc_start + 118):(loc_start + 122)].view(dtype =  "f4")[0]
        ch_info["kind"]             = file[(loc_start + 122):(loc_start + 123)].view(dtype = "<u1")[0]
        ch_info["unused1"]          = file[(loc_start + 123):(loc_start + 124)].view(dtype = "<i1")[0]
        ch_info["scale"]            = file[(loc_start + 124):(loc_start + 128)].view(dtype =  "f4")[0]
        ch_info["offset"]           = file[(loc_start + 128):(loc_start + 132)].view(dtype =  "f4")[0]
        ch_info["unit"]             = file[(loc_start + 132):(loc_start + 138)].view(dtype =  "S6")[0]
        ch_info["idx"]              = ch_idx
    
        if (ch_info["firstblock"] == -1):
            continue
        
        ch_infos.append(ch_info)
    
    file_info["channels"] = len(ch_infos)
    
    return (file_info, ch_infos)

def read_block_hdr(file, start_block):
    block_info = dict()
    
    block_info["pred_block"]        = file[(start_block + 0):(start_block + 4)].view(dtype = "i4")[0]
    block_info["succ_block"]        = file[(start_block + 4):(start_block + 8)].view(dtype = "i4")[0]
    block_info["start_time"]        = file[(start_block + 8):(start_block + 12)].view(dtype = "i4")[0]
    block_info["end_time"]          = file[(start_block + 12):(start_block + 16)].view(dtype = "i4")[0]
    block_info["channel_num"]       = file[(start_block + 16):(start_block + 18)].view(dtype = "i2")[0]
    block_info["items"]             = file[(start_block + 18):(start_block + 20)].view(dtype = "i2")[0]
    
    return block_info

def read_analog_signal(file, start_block, block_info):
    return file[(start_block + 20):(start_block + int(block_info["items"] * 2 + 20))].view(dtype = "i2")

def read_wavmks(start_block, ch_infos, ch_idx, block_info, file, file_info, spike_data):
    fs = 1/(ch_infos[-1]["l_chan_dvd"]*file_info["dtime_base"]*file_info["us_per_time"])
    
    loc_start = start_block + 20
    
    wave_info = dict()
    wave_info["spike_sz"] = ch_infos[ch_idx]["n_extra"]
    wave_info["header_sz"] = 8
    wave_info["wavmk_sz"] = wave_info["header_sz"] + wave_info["spike_sz"]  
    
    #===============================================================
    # plt.figure()
    #===============================================================
    for spike_idx in range(block_info["items"]):
        
        start = loc_start + 0 + int(spike_idx * (wave_info["wavmk_sz"]))
        start_time = file[start:(start + 4)].view(dtype = "<u4")[0] * file_info["dtime_base"] * fs
                
        start = loc_start + wave_info["header_sz"] + int(spike_idx * (wave_info["wavmk_sz"]))
        spike = file[start:(start + wave_info["spike_sz"])].view(dtype = "<i2")
        
        #print(start_time/fs, spike_data.shape)
        if (int(start_time + wave_info["spike_sz"]/2) < len(spike_data)):
            spike_data[int(start_time):(int(start_time + wave_info["spike_sz"]/2))] = spike
    #===============================================================
    #     plt.plot(spike)
    # plt.show(block = True)
    #===============================================================
    
    #===============================================================
    # plt.figure()
    # offset = start_block + 20
    # for x in range(2,100,2):
    #     plt.plot(file[(offset + 8):(offset + 8 + x)].view(dtype = "i2"))
    #     plt.show(block = False)
    #     print(x)
    #     plt.pause(2)
    #     plt.clf()
    # plt.close()
    #===============================================================
    
    
    return spike_data

#-- def read_wavmks(start_block, ch_infos, ch_idx, block_info, file, file_info):
    # fs = 1/(ch_infos[-1]["l_chan_dvd"]*file_info["dtime_base"]*file_info["us_per_time"])
    # spike_data = np.zeros((int(file_info["max_ftime"] * file_info["dtime_base"] * fs,)))
#------------------------------------------------------------------------------ 
    #---------------------------------------------- loc_start = start_block + 20
#------------------------------------------------------------------------------ 
    #-------------------------------------------------------- wave_info = dict()
    #----------------------- wave_info["spike_sz"] = ch_infos[ch_idx]["n_extra"]
    #------------------------------------------------ wave_info["header_sz"] = 8
    #---- wave_info["wavmk_sz"] = wave_info["header_sz"] + wave_info["spike_sz"]
#------------------------------------------------------------------------------ 
    #---------- #===============================================================
    #------------------------------------------------------------ # plt.figure()
    #---------- #===============================================================
    #------------------------------ for spike_idx in range(block_info["items"]):
#------------------------------------------------------------------------------ 
        #-------------------------------------------------- if (spike_idx == 5):
            #-------------------------------------------------------- print("A")
#------------------------------------------------------------------------------ 
        #------ start = loc_start + 0 + int(spike_idx * (wave_info["wavmk_sz"]))
        # start_time = file[start:(start + 4)].view(dtype = "<u4")[0] * file_info["dtime_base"] * fs
#------------------------------------------------------------------------------ 
        # start = loc_start + wave_info["header_sz"] + int(spike_idx * (wave_info["wavmk_sz"]))
        # spike = file[start:(start + wave_info["spike_sz"])].view(dtype = "<i2")
#------------------------------------------------------------------------------ 
        #-------------------------------- print(start_time/fs, spike_data.shape)
        #----- if (int(start_time + wave_info["spike_sz"]/2) < len(spike_data)):
            # spike_data[int(start_time):(int(start_time + wave_info["spike_sz"]/2))] = spike
    #---------- #===============================================================
    #----------------------------------------------------- #     plt.plot(spike)
    #-------------------------------------------------- # plt.show(block = True)
    #---------- #===============================================================
#------------------------------------------------------------------------------ 
    #---------- #===============================================================
    #------------------------------------------------------------ # plt.figure()
    #----------------------------------------------- # offset = start_block + 20
    #------------------------------------------------ # for x in range(2,100,2):
    #---- #     plt.plot(file[(offset + 8):(offset + 8 + x)].view(dtype = "i2"))
    #--------------------------------------------- #     plt.show(block = False)
    #------------------------------------------------------------ #     print(x)
    #-------------------------------------------------------- #     plt.pause(2)
    #----------------------------------------------------------- #     plt.clf()
    #------------------------------------------------------------- # plt.close()
    #---------- #===============================================================
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #------------------------------------------------ return spike_data.tolist()

def read_data(file, file_info, ch_infos):
    
    ch_data = list()
    for ch_idx in range(file_info["channels"]):
        if (ch_infos[ch_idx]["firstblock"] == -1):
            continue
        ch_data.append(list())
        
        start_block = ch_infos[ch_idx]["firstblock"]
        if (ch_infos[ch_idx]["kind"] == 1):
            for _ in range(ch_infos[ch_idx]["blocks"]):
                block_info = read_block_hdr(file, start_block)                
                ch_data[-1].extend(read_analog_signal(file, start_block, block_info))
                
                start_block = block_info["succ_block"]
        
        fs = 1/(ch_infos[-1]["l_chan_dvd"]*file_info["dtime_base"]*file_info["us_per_time"])
        spike_data = np.zeros((int(file_info["max_ftime"] * file_info["dtime_base"] * fs,)))
        if (ch_infos[ch_idx]["kind"] == 6):
            for _ in range(ch_infos[ch_idx]["blocks"]):
                block_info = read_block_hdr(file, start_block)
                #Header is 8 bytes [4 startbytes, 4 pattern_id_bytes]
                spike_data = read_wavmks(start_block, ch_infos, ch_idx, block_info, file, file_info, spike_data)
                
                
                start_block = block_info["succ_block"]
            spike_data = spike_data.tolist()
            ch_data[-1].extend(spike_data)
        
        ch_data[-1] = np.asarray(ch_data[-1], dtype = np.float32)
        ch_data[-1] *= ch_infos[ch_idx]["scale"] / magic_numbers["scale_multiplier"]
        ch_data[-1] += ch_infos[ch_idx]["offset"]
        
    return ch_data

#------------------------------------- def read_data(file, file_info, ch_infos):
#------------------------------------------------------------------------------ 
    #---------------------------------------------------------- ch_data = list()
    #------------------------------- for ch_idx in range(file_info["channels"]):
        #---------------------------- if (ch_infos[ch_idx]["firstblock"] == -1):
            #---------------------------------------------------------- continue
        #------------------------------------------------ ch_data.append(list())
#------------------------------------------------------------------------------ 
        #-------------------------- start_block = ch_infos[ch_idx]["firstblock"]
        #--------------------------- for _ in range(ch_infos[ch_idx]["blocks"]):
            #-------------------- block_info = read_block_hdr(file, start_block)
#------------------------------------------------------------------------------ 
            #------------------------------- if (ch_infos[ch_idx]["kind"] == 1):
                # ch_data[-1].extend(read_analog_signal(file, start_block, block_info))
#------------------------------------------------------------------------------ 
            #------------- #Header is 8 bytes [4 startbytes, 4 pattern_id_bytes]
            #------------------------------- if (ch_infos[ch_idx]["kind"] == 6):
                # ch_data[-1].extend(read_wavmks(start_block, ch_infos, ch_idx, block_info, file, file_info))
#------------------------------------------------------------------------------ 
            #---------------------------- start_block = block_info["succ_block"]
#------------------------------------------------------------------------------ 
        #------------- ch_data[-1] = np.asarray(ch_data[-1], dtype = np.float32)
        # ch_data[-1] *= ch_infos[ch_idx]["scale"] / magic_numbers["scale_multiplier"]
        #----------------------------- ch_data[-1] += ch_infos[ch_idx]["offset"]
#------------------------------------------------------------------------------ 
    #------------------------------------------------------------ return ch_data

def read_file(file_path):
    file = np.memmap(file_path, dtype='u1', offset=0, mode='r')
    file_info = read_file_info(file)
    (file_info, ch_infos) = read_channel_header(file, file_info)
    data = read_data(file, file_info, ch_infos)
    
    return (file_info, ch_infos, data)

def test():
    filePath = "/home/voodoocode/Downloads/2891/tmp4.smr"    
    filePath = "/home/voodoocode/Downloads/2891/2851_s1_635_power_pinch.smr"
    
    (file_info, ch_infos, data) = read_file(filePath)
    
    ch_cnt = len(data)
    (fig, axes) = plt.subplots(ch_cnt, 1)
    for idx in range(ch_cnt):
        axes[idx].plot(data[idx])
        print(ch_infos[idx])
        
    plt.show(block = True)
    
    print("Terminated successfully")
    quit()

def __mainInner(basePath, file, acc_channel, overwrite):
    filePath = basePath + file + ".smr"
    
    #------------------------------------------------ if (file != "642-2188-a"):
        #---------------------------------------------------------------- return
    
    if (os.path.exists(filePath + "_conv_hdr.pkl") == True):
        return
    
    print(file)
    
    filePath = "/mnt/data/Professional/UHN/pac_investigation/data/beta/3137-beta-2.smr"
    filePath = "/home/voodoocode/Downloads/2891/2851_s1_635_power_pinch.smr"
    #filePath = "/home/voodoocode/Downloads/2891/tmp2.smr"
    #filePath = "/home/voodoocode/Downloads/2891/tmp3.smr"
    filePath = "/home/voodoocode/Downloads/2891/tmp4.smr"
    #filePath = "/home/voodoocode/Downloads/2891/tmp5.smr"

    file = neo.Spike2IO(filePath)
    data2 = np.asarray(file.read(lazy=False, load_waveforms=True)[0].segments[0].analogsignals[0])[:, 1]

    file = np.memmap(filePath, dtype='u1', offset=0, mode='r')
    file_info = read_file_info(file)
    (file_info, ch_infos) = read_channel_header(file, file_info)
    data = read_data(file, file_info, ch_infos)
    
    data1 = np.copy(data[2]); data1 -= np.min(data1); data1 = data1/np.max(data1)

    file = neo.Spike2IO(filePath)
    data2 = np.copy(data2); data2 -= np.min(data2); data2 = data2/np.max(data2)
    
    #===========================================================================
    # print(len(data2))
    # print(len(data1), ch_infos[2]["ideal_rate"], ch_infos[2]["idx"])
    # 
    # plt.plot(data2)
    # plt.plot(data1)
    #===========================================================================
    
    ch_cnt = len(data)
    (fig, axes) = plt.subplots(ch_cnt, 1)
    for idx in range(ch_cnt):
        axes[idx].plot(data[idx])
        print(ch_infos[idx])
        
    #===========================================================================
    # plt.figure()
    # plt.plot(np.memmap(filePath, dtype='u1', offset=0, mode='r').view(dtype = "i2"))
    #===========================================================================
        
    plt.show(block = True)
    
    print("Terminated successfully")
    quit()
    
    raw_data = [np.asarray(file.read(lazy=False)[0].segments[0].analogsignals[0])[:, 0]]
    
    if (len(file.read(lazy=True)[0].segments[0].analogsignals) == 1):
        fs = [file.get_signal_sampling_rate([0])]
        raw_data = [np.asarray(file.read(lazy=False)[0].segments[0].analogsignals[0])[:, 0]]
        
        hdr = {20 : {"fs" : fs[0]}}
        data = {20 : raw_data[0]}
    elif (len(file.read(lazy=True)[0].segments[0].analogsignals) == 2):
        fs = [file.get_signal_sampling_rate([0]), file.get_signal_sampling_rate([1])]
        raw_data = [np.asarray(file.read(lazy=False)[0].segments[0].analogsignals[0])[:, 0], 
                    np.asarray(file.read(lazy=False)[0].segments[0].analogsignals[1])[:, 0]]
    
        if (len(raw_data[0]) > len(raw_data[1])):
            data = {20 : raw_data[0], 21 : raw_data[1]}
        else:
            data = {20 : raw_data[1], 21 : raw_data[0]}
        if (fs[0] > fs[1]):
            hdr = {20 : {"fs" : fs[0]}, 21 : {"fs" : fs[1]}}
        else:
            hdr = {20 : {"fs" : fs[1]}, 21 : {"fs" : fs[0]}}
    else:
        raise AssertionError
    
    filePath = filePath[:filePath.index(".smr")]
    pickle.dump(hdr, open(filePath + ".txt_conv_hdr.pkl", "wb"))
    pickle.dump(data, open(filePath + ".txt_conv_data.pkl", "wb"))


test()
main("/mnt/data/Professional/UHN/pac_investigation/data/tremor/meta.ods",
     "/mnt/data/Professional/UHN/pac_investigation/data/tremor/", 4, False)
print("Terminated successfully")

















