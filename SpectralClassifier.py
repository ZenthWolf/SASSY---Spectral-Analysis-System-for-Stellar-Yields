# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:13:07 2023

@author: Zenth
"""

from astropy.table import Table
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DataListDir = 'SDSS_Data_Subset/'
DataFileDir = DataListDir + 'Spectra/'

trainingTable = Table.read(DataListDir + 'Training_Data.fits')
testingTable = Table.read(DataListDir + 'Testing_Data.fits')

# Original Survey
fluxMax = 252.4149169921875
lmbMin = 3552.2204983897977
lmbMax = 10399.206408133097

# From current data set
# fluxMax = 3460.751708984375
# lmbMin = 3553.0396325911483
# lmbMax = 10387.235932988217

spectralSizeMin = 2889
spectralSizeMax = 4631

'''
for i in range(len(trainingTable)):
    if i%100 == 0:
        print(f'{i} / {len(trainingTable)}')
    samplestar = Table(trainingTable[i])
    samplename = samplestar['specobjid'][0]
    with fits.open(DataFileDir + f'{samplename}.fits') as hdu:
        spectrum = hdu[1].data
    testlen = len(spectrum['flux'])
    spectralSizeMax = max(spectralSizeMax, testlen)
    spectralSizeMin = min(spectralSizeMin, testlen)

# 2889
# 4631
print(spectralSizeMin)
print(spectralSizeMax)

for i in range(len(testingTable)):
    if i%100 == 0:
        print(f'{i} / {len(testingTable)}')
    samplestar = Table(testingTable[i])
    samplename = samplestar['specobjid'][0]
    with fits.open(DataFileDir + f'{samplename}.fits') as hdu:
        spectrum = hdu[1].data
    testlen = len(spectrum['flux'])
    spectralSizeMax = max(spectralSizeMax, testlen)
    spectralSizeMin = min(spectralSizeMin, testlen)

print(spectralSizeMin)
print(spectralSizeMax)
'''
#tf.keras.layers.Conv1D(32, 10, activation='relu'),

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class_mapping = {
    'O': np.array([0]),
    'B': np.array([1]),
    'A': np.array([2]),
    'F': np.array([3]),
    'G': np.array([4]),
    'K': np.array([5]),
    'M': np.array([6])
}

for i in range(int(len(trainingTable))):
    trainingStar = Table(trainingTable[i])
    starName = trainingStar['specobjid'][0]
    with fits.open(DataFileDir + f'{starName}.fits') as hdu:
        spectrum = hdu[1].data
    # Normalize
    flux = spectrum['flux']/fluxMax
    flux = np.clip(flux, 0, np.inf)
    wavelength = (10 ** spectrum['loglam'] - lmbMin)/lmbMax
    
    # Smooth
    window_size = 50
    smoothed_flux = np.convolve(flux, np.ones(window_size)/window_size, mode='same')
    
    # plt.plot(wavelength, smoothed_flux)
    # plt.xlabel('Wavelength [$\AA$]')
    # plt.ylabel('Flux [10$^{-17}$ erg/cm$^2$/s/$\AA$]')
    # plt.show()
    
    if (zeros := spectralSizeMax - len(flux)) > 0:
        zero_array = np.zeros(zeros)
        flux = np.append(flux, zero_array)
        wavelength = np.append(wavelength, zero_array)
    
    sample_data = np.array([np.concatenate((flux[:, np.newaxis], wavelength[:, np.newaxis]), axis=1)])
    sample_class = class_mapping[trainingStar['subclass'][0][0]]
    
    if (i==0):
        input_data = sample_data
        ytrain = sample_class
    else:
        input_data = np.concatenate((input_data, sample_data), axis=0)
        ytrain = np.concatenate((ytrain, sample_class), axis=0)
    #prediction = model.predict(input_data)
    
    #if(i%10 == 0):
    #    print('')
    #    print('Modelled as:')
    #    print(prediction)
    #    predicted_index = np.argmax(prediction)
    #    predicted_class = list(class_mapping.keys())[predicted_index]
    #    print(f'Predicted: {predicted_class}: with {np.max(prediction)} certainty')
    #    print(f"Actual: {trainingStar['subclass'][0][0]}")
    #    print('\n\n')
    
model.fit(input_data, ytrain, epochs=20, shuffle=True)

for i in range(len(testingTable)):
    if i%100 == 0:
        print(i)
    testingStar = Table(testingTable[i])
    starName = testingStar['specobjid'][0]
    with fits.open(DataFileDir + f'{starName}.fits') as hdu:
        spectrum = hdu[1].data
    
    # Normalize
    flux = spectrum['flux']/fluxMax
    flux = np.clip(flux, 0, np.inf)
    wavelength = (10 ** spectrum['loglam'] - lmbMin)/lmbMax
    
    # Smooth
    window_size = 50
    smoothed_flux = np.convolve(flux, np.ones(window_size)/window_size, mode='same')
    
    # plt.plot(wavelength, smoothed_flux)
    # plt.xlabel('Wavelength [$\AA$]')
    # plt.ylabel('Flux [10$^{-17}$ erg/cm$^2$/s/$\AA$]')
    # plt.show()
    
    if (zeros := spectralSizeMax - len(flux)) > 0:
        zero_array = np.zeros(zeros)
        flux = np.append(flux, zero_array)
        wavelength = np.append(wavelength, zero_array)
    
    sample_data = np.array([np.concatenate((flux[:, np.newaxis], wavelength[:, np.newaxis]), axis=1)])
    sample_class = class_mapping[testingStar['subclass'][0][0]]
    
    if (i==0):
        input_data = sample_data
        ytest = sample_class
    else:
        input_data = np.concatenate((input_data, sample_data), axis=0)
        ytest = np.concatenate((ytest, sample_class), axis=0)

model.evaluate(input_data, ytest, verbose=2)

# Loss, accuracy

print("\nNo Conv Layer:")
print("20 epochs:")
print("WindowSize 200:")
print("(300, 200) -> [0.8667, 0.6207]")
print("WindowSize 50:")
print("(300, 200) -> [0.9085, 0.5793]")

print("\n 5 Epochs:")
print("(100, 250) -> [1.38, 0.43999]")
print("(200, 250) -> [1.18, 0.5486]")
print("(300, 150) + ws200 -> [1.0841, 0.5871]")
print("(300, 150) + ws50 -> [1.1121, 0.5593]")
print("(300, 150) + ws10 -> [1.1060, 0.5793]")
print("(300, 150) + ws1 -> [1.1566, 0.5186]")
print("(300, 150, 100) -> [1.1057,0.5529]")
print("(450)      -> [1.2499, 0.4486]>")
print("(150, 150, 150) -> (1.2533, 0.4857)")
print("(200, 150, 100) -> (1.2569, 0.5507)")

print("\nWith Conv Layer:")
print("Conv 32, 10")
print("Window 200")
print("(300, 200) -> (0.6864, 0.7164)")


# model.save('model.h5')
# loaded_model = tf.keras.models.load_model('model.h5')
# OR
# model.load_weights('my_model_weights.h5')
