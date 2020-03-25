# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:26:28 2020

@author: Camila
"""

import os
import mne, re 

import pandas as pd
import numpy.fft as fft
import numpy as np



path = 'C:/Users/Camila/Documents/Tesis/raw_data/healthy/'

# valida la extension del fichero
def is_edf_file(file_name):
    return file_name[len(file_name)-3:len(file_name)] == "edf" 


# revisa si es un canal valido
match_regex = lambda channel_name : re.compile(r'(P|F|O|T)').search(channel_name)

# valida la extension del fichero
is_edf_file = lambda file_name : file_name[len(file_name)-3:len(file_name)] == "edf"


files = os.listdir(path)

edf_files = [name for name in files if is_edf_file(name)]
print(edf_files)

def extraer_canal(datos, nombre_canal):

    canal = pd.DataFrame(datos[
        datos.ch_names.index(nombre_canal)
    ][0])
    valores_canal = canal.T
    valores_canal.describe()
    datos = valores_canal.values

    return datos

# bands limits dictionary
bands = {
        'alpha' : {
                'low': 8 ,
                'high': 13
                },
        'betha' : {
                'low': 13,
                'high': 30
                },
        'delta' : {
                'low': 0,
                'high': 4
                },
        'gamma' : {
                'low': 30,
                'high': 1000
                },
        'theta' : {
                'low': 4,
                'high': 8
                },
        }
        
channels_data = []        
for file_name in edf_files:

    datos_edf = mne.io.read_raw_edf(path+file_name, preload=True, stim_channel=None)

    list_channels = list(
        filter(
            lambda channel_name : len(channel_name) > 4 and channel_name[0:3] == "EEG", 
            datos_edf.info["ch_names"]
        )
    )
    
    list_channels = [channel for channel in list_channels if match_regex(channel[4])]
    fs = datos_edf.info['sfreq']
    
    
    
    # iterate over each valid channel on the edf
    for channel in list_channels:
        df_channel = pd.DataFrame()
        df_channel['exam'] = [file_name]
        df_channel['channel'] = [channel]
        df_channel['class'] = [0]
        
        arr_canal = extraer_canal(datos_edf, channel)
        arr_canal = arr_canal.ravel()
        
        # welch periodogram computations
        win = 4 * fs
        freqs, psd = signal.welch(arr_canal, fs, nperseg=win)
        for i in bands:
            idx_band = np.logical_and(freqs >= bands[i]['low'], freqs <= bands[i]['high'])
            freq_res = freqs[1] - freqs[0]
            
            # integration
            band_power = simps(psd[idx_band], dx=freq_res)
            total_power = simps(psd, dx=freq_res)
            
            perc_power = band_power / total_power
            
            df_channel[i] = [perc_power]
        
        channels_data.append(df_channel)
    
    # generating a df per exam and exporting data to csv    
    merged = pd.concat(channels_data, ignore_index=True)    
    
 merged.to_csv("healthy.csv",index=False)
