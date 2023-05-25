# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:30:00 2023

@author: Zenth
"""

import logging
import time
from urllib.error import URLError
from socket import timeout as TimeoutError

from astroquery.sdss import SDSS
from astropy.table import Table, vstack
from astropy.io import fits

import numpy as np

import os
print(os.getcwd())

DataListDir = 'SDSS_Data_Subset/'
DataFileDir = DataListDir + 'Spectra/'

logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def QueryStars():
    # Perform the query and retrieve a random sample of spectra and their corresponding classes
    querybase = "SELECT TOP 3000 s.specobjid, s.mjd, s.fiberID, s.class, s.subclass, s.run2d, s.plate FROM specobj s WHERE s.class='star' AND s.subclass LIKE "
    
    query = querybase + "'O%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassO_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'B%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassB_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'A%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassA_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'F%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassF_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'G%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassG_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'K%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassK_Data_Annex.fits', format='fits', overwrite=True)
    
    time.sleep(50)
    
    query = querybase + "'M%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassM_Data_Annex.fits', format='fits', overwrite=True)


# Time padding required to prevent overloading servers
# Still seems to be too frequent for the number of requests
# Consider 30 s between each (and toning down between classes) if running again

def Wait(minutes):
    print(f'Retrying in {minutes} minutes')
    if minutes > 5:
        if smoothing_time := minutes%5 != 0:
            time.sleep(smoothing_time * 60)
            Wait(minutes-smoothing_time)
        else:
            time.sleep(300)
            Wait(minutes-5)
    else:
        time.sleep(60)
        Wait(minutes-1)

def GetSpectra(star):
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        try:
            return SDSS.get_spectra(matches=star)
        
        except (URLError, TimeoutError) as e:
            error_message = "An error occurred: " + str(e)
            print(error_message)
            logging.error(error_message)
            attempts += 1
            
            if attempts < max_attempts:
                Wait(5 + 5*attempts)
            else:
                logging.error("Maximum Retry Failures")
                print("Maximum retry attempts.")
                raise

def DownloadData(star_class, maxIndex = -1):
    MnemonicDict = {
        'O': 'Oh,',
        'B': 'Be',
        'A': 'A',
        'F': 'Fine',
        'G': 'Girl',
        'K': 'Kiss',
        'M': 'Me'
        }
    
    dataset = Table.read(DataListDir + f'Class{star_class}_Data_Annex.fits')
    print(MnemonicDict[star_class])
    print(len(dataset))
    
    if maxIndex == -1:
        maxIndex = len(dataset)
    
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class {star_class}')
            continue
        data = GetSpectra(samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=False)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class {star_class}')
        time.sleep(20)
    
    print(MnemonicDict[star_class] + ' completed')

def StoreSpectralData():
    maxIndex = 2000
    stellar_class = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    logging.info(f'Spectral Data to download: {maxIndex} per class')
    
    for star_class in stellar_class:
        DownloadData(star_class, maxIndex)
        time.sleep(600)
'''
    print('Oh,')
    dataset = results_o
    print(len(dataset))
    for i in range(maxIndex):
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class O')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class O')
        time.sleep(20)
    
    print('Oh, completed')
#    time.sleep(600)
    
    print('Be')
    dataset = results_b
    print(len(dataset))
    for i in range(maxIndex):
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class B')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class B')
        time.sleep(20)
    print('Be completed')
#    time.sleep(600)
    
    print('A')
    dataset = results_a
    print(len(dataset))
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class A')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class A')
        time.sleep(20)
    
    print('A completed')
#    time.sleep(600)
    
    print('Fine')
    dataset = results_f
    print(len(dataset))
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class F')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class F')
        time.sleep(20)
    
    print('Fine completed')
#    time.sleep(600)
    
    print('Girl')
    dataset = results_g
    print(len(dataset))
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class G')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class G')
        time.sleep(20)
    
    print('Girl completed')
#    time.sleep(600)
    
    print('Kiss')
    dataset = results_k
    print(len(dataset))
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class K')
            continue
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class K')
        time.sleep(20)
    
    print('Kiss Completed')
    time.sleep(600)
    
    print('Me')
    dataset = results_m
    print(len(dataset))
    for i in range(maxIndex):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        candidate = DataFileDir + f'{samplename}.fits'
        if os.path.isfile(candidate):
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class M')
            continue
        data = GetSpectra(samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=True)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class M')
        time.sleep(20)
    
    print('Me completed')
'''

def GetDataBounds():
    results_o = Table.read(DataListDir + 'ClassO_Data.fits')
    results_b = Table.read(DataListDir + 'ClassB_Data.fits')
    results_a = Table.read(DataListDir + 'ClassA_Data.fits')
    results_f = Table.read(DataListDir + 'ClassF_Data.fits')
    results_g = Table.read(DataListDir + 'ClassG_Data.fits')
    results_k = Table.read(DataListDir + 'ClassK_Data.fits')
    results_m = Table.read(DataListDir + 'ClassM_Data.fits')
    
    fluxMax = 0
    lmbMin = np.inf
    lmbMax = -np.inf
    
    print('Oh,')
    dataset = results_o
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('Be')
    dataset = results_b
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('A')
    dataset = results_a
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('Fine')
    dataset = results_f
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('Girl')
    dataset = results_g
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('Kiss')
    dataset = results_k
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print('Me')
    dataset = results_m
    print(len(dataset))
    
    for i in range(len(dataset)):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    print(f'fluxMax = {fluxMax}')
    print(f'lmbMin = {lmbMin}')
    print(f'lmbMax = {lmbMax}')
    
    # fluxMax = 3460.751708984375
    # lmbMin = 3553.0396325911483
    # lmbMax = 10387.235932988217

def PrepDataIndexing (table):
    table['subclass'] = table['subclass'].astype(str)
    array = table[['specobjid', 'subclass']].as_array()
    np.random.shuffle(array)
    return Table(array)


def PartitionTrainingData():
    results_o = Table.read(DataListDir + 'ClassO_Data.fits')
    results_b = Table.read(DataListDir + 'ClassB_Data.fits')
    results_a = Table.read(DataListDir + 'ClassA_Data.fits')
    results_f = Table.read(DataListDir + 'ClassF_Data.fits')
    results_g = Table.read(DataListDir + 'ClassG_Data.fits')
    results_k = Table.read(DataListDir + 'ClassK_Data.fits')
    results_m = Table.read(DataListDir + 'ClassM_Data.fits')
    
    results_o = PrepDataIndexing(results_o)
    results_b = PrepDataIndexing(results_b)
    results_a = PrepDataIndexing(results_a)
    results_f = PrepDataIndexing(results_f)
    results_g = PrepDataIndexing(results_g)
    results_k = PrepDataIndexing(results_k)
    results_m = PrepDataIndexing(results_m)
    
    trainingdata = vstack([results_o[['specobjid', 'subclass']][:800], results_b[['specobjid', 'subclass']][:800], 
                           results_a[['specobjid', 'subclass']][:800], results_f[['specobjid', 'subclass']][:800], 
                           results_g[['specobjid', 'subclass']][:800], results_k[['specobjid', 'subclass']][:800], 
                           results_m[['specobjid', 'subclass']][:800]
                         ])
    trainingdata = PrepDataIndexing(trainingdata)
    
    testingdata = vstack([results_o[['specobjid', 'subclass']][-200:], results_b[['specobjid', 'subclass']][-200:], 
                          results_a[['specobjid', 'subclass']][-200:], results_f[['specobjid', 'subclass']][-200:], 
                          results_g[['specobjid', 'subclass']][-200:], results_k[['specobjid', 'subclass']][-200:], 
                          results_m[['specobjid', 'subclass']][-200:]
                        ])
    testingdata = PrepDataIndexing(testingdata)
    
    trainingdata.write(DataListDir + 'Training_Data.fits', format='fits', overwrite=True)
    testingdata.write(DataListDir + 'Testing_Data.fits', format='fits', overwrite=True)

### Disabled ### QueryStars()

### Disabled ### StoreSpectralData()

### Disabled ### GetDataBounds()

### Disabled ### PartitionTrainingData()

print('success')