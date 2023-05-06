# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:13:07 2023

@author: Zenth
"""

from astroquery.sdss import SDSS
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DataListDir = 'SDSS_Data_Subset/'

table = Table.read(DataListDir + 'ClassK_Data.fits')
data = SDSS.get_spectra(matches=table)

print('Data Read')

# Original Survey
fluxMax = 252.4149169921875
lmbMin = 3552.2204983897977
lmbMax = 10399.206408133097

# From current data set
# fluxMax = 3460.751708984375
# lmbMin = 3553.0396325911483
# lmbMax = 10387.235932988217

chosenSample = 1

spectrum = data[chosenSample]
# Normalize
flux = spectrum[1].data['flux']/fluxMax
flux = np.clip(flux, 0, np.inf)
wavelength = (10 ** spectrum[1].data['loglam'] - lmbMin)/lmbMax

print('Pre-Smooth')

# Smooth
window_size = 10
smoothed_flux = np.convolve(flux, np.ones(window_size)/window_size, mode='same')

print('Smoothed')

plt.plot(wavelength, smoothed_flux)

plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Flux [10$^{-17}$ erg/cm$^2$/s/$\AA$]')
plt.show()

spectralSizeMin = 3800
spectralSizeMax = 4627

for i in range(1):
    testlen = len(data[i][1].data['flux'])
    spectralSizeMax = max(spectralSizeMax, testlen)
    spectralSizeMin = min(spectralSizeMin, testlen)

print(spectralSizeMin)
print(spectralSizeMax)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])

# model.save('model.h5')
# loaded_model = tf.keras.models.load_model('model.h5')
# OR
# model.load_weights('my_model_weights.h5')

input_data = np.array([np.concatenate((flux[:, np.newaxis], wavelength[:, np.newaxis]), axis=1)])
input_data = input_data[:, :3819, :]

input_data = np.concatenate((input_data, input_data), axis=0)

print('len')
print(len(input_data[0]))

output_data = model.predict(input_data)



print('outputted')

print(table[chosenSample])
print('Modelled as:')
print(output_data)
