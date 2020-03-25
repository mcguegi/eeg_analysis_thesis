# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:36:16 2019

@author: Camila
"""

import mne
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------------------

# read edf file and get basic info
datos_edfN = mne.io.read_raw_edf('C:/Users/Camila/Documents/Tesis/raw_data/healthy/00010501_s001_t001.edf', preload=True, stim_channel=None)
datos_edfN.info
datos_edfN.rename_channels(lambda s: s.strip("EEG").strip('-REF').replace(' ',''))
datos_edfN.info['ch_names']
datos_edfN.set_eeg_reference("average")

# plot the channels of edf
datos_edfN.plot(block=True, lowpass=3)
datos_edfN.plot(n_channels=5 , lowpass=3)

# get the sample frequency
sf = datos_edfN.info['sfreq']

# looking for indexes of channels
datos_edfN.info['ch_names'].index('F7')
datos_edfN.info['ch_names'].index('T4')
datos_edfN.info['ch_names'].index('F3')
datos_edfN.info['ch_names'].index('C3')
datos_edfN.info['ch_names'].index('P3')

# extracting info of channels and saving them into txt for MATLAB
F7 = datos_edfN[10][0].T
np.savetxt(r'C:/Users/Camila/plain_files_eeg/E2-F7.txt', F7, fmt='%10.30f', delimiter="\n")

T4 = datos_edfN[13][0].T
np.savetxt(r'C:/Users/Camila/plain_files_eeg/'+'E2-T4.txt', T4, fmt='%10.30f', delimiter="\n")

F3 = datos_edfN[2][0].T
np.savetxt(r'C:/Users/Camila/plain_files_eeg/'+'E2-F3.txt', F3, fmt='%10.30f', delimiter="\n")

C3 = datos_edfN[4][0].T
np.savetxt(r'C:/Users/Camila/plain_files_eeg/'+'E2-C3.txt', C3, fmt='%10.30f', delimiter="\n")

P3 = datos_edfN[6][0].T
np.savetxt(r'C:/Users/Camila/plain_files_eeg/'+'E2-P3.txt', P3, fmt='%10.30f', delimiter="\n")



