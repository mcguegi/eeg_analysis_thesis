# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:27:36 2020

@author: Camila
"""


import csv
import re
import mne
import matplotlib.pyplot as plt
import numpy as np
import os

basepath = './examenes/enfermos'

list_examenes = []
real_list_channels = ['EEG FP1-REF','EEG FP2-REF','EEG F3-REF','EEG F4-REF','EEG C3-REF','EEG C4-REF','EEG P3-REF','EEG P4-REF','EEG O1-REF','EEG O2-REF','EEG F7-REF','EEG F8-REF','EEG T3-REF','EEG T4-REF','EEG T5-REF','EEG T6-REF','EEG A1-REF','EEG A2-REF','EEG FZ-REF','EEG CZ-REF','EEG PZ-REF']
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)) and entry[-4:] == '.edf':
        datos_edf = mne.io.read_raw_edf(
            os.path.join(basepath, entry), 
            preload=True, 
            stim_channel=None
        )
        list_examenes.append(
            (entry[:-4], datos_edf)
        )



def get_data_to_csv(datos_edf):


    list_channels = list(datos_edf.info["ch_names"])
    list_channels = list(set(list_channels) - (set(list_channels) - set(real_list_channels)))
    list_channels = [channel for channel in list_channels]
    get_data_channel = lambda channel_name : datos_edf[datos_edf.ch_names.index(channel)][0][0]
    freq_bands_list = []

    for channel in list_channels:
        
        current_data = get_data_channel(channel)
        channel_f = np.fft.fft(current_data)
        sampling_freq = datos_edf.info['sfreq']
        len_channel = len(channel_f)
        k_array = [i for i in range(0, len_channel)]
        f_values = [((sampling_freq*i)/len_channel) for i in k_array]

        channel_f = abs(channel_f)
        get_index = lambda x : f_values.index(x)
        
        band_width = {
            "channel": channel,
            "delta" : np.mean(channel_f[0:get_index(4)]),
            "tetha" : np.mean(channel_f[get_index(4):get_index(8)]),
            "alpha" : np.mean(channel_f[get_index(8):get_index(13)]),
            "beta" : np.mean(channel_f[get_index(13):get_index(30)]),
            "gamma" : np.mean(channel_f[get_index(30):get_index(60)]),
            "class" : 1
        }
        band_width = [v*10 if k!='class' else v for k,v in band_width.items()]
        freq_bands_list.append(band_width)
    return freq_bands_list
    

for (name, data) in list_examenes:
    with open(name+'.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['channel', 'delta','tetha','alpha','beta','gamma','class'])
        data_freq_list = get_data_to_csv(data)
        for i in range(len(data_freq_list)):
            writer.writerow(data_freq_list[i])