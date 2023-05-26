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

def QueryStellarClass(stellar_class):
    # Perform the query and retrieve a random sample of spectra and their corresponding classes
    querybase = "SELECT TOP 3000 s.specobjid, s.mjd, s.fiberID, s.class, s.subclass, s.run2d, s.plate FROM specobj s WHERE s.class='star' AND s.subclass LIKE "
    query = querybase + f"'{stellar_class}%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'Class{stellar_class}_Data_Annex.fits', format='fits', overwrite=True)
    

def QueryStars():
    stellar_class = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    for s in stellar_class:
        QueryStellarClass(s)
        time.sleep(60)

# Time padding required to prevent overloading servers
# Watch for failure and attempt with delay 2 additional times

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
        if (remaining_time := minutes-1) > 0:
            Wait(remaining_time)

def GetSpectra(star):
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        try:
            return SDSS.get_spectra(matches=star)
        
        except (URLError, TimeoutError, ConnectionResetError) as e:
            error_message = "An error occurred: " + str(e)
            print(error_message)
            logging.error(error_message)
            attempts += 1
            
            if attempts < max_attempts:
                if isinstance(e, ConnectionResetError):
                    Wait(5)
                Wait(5 + 5*attempts)
            else:
                logging.error("Maximum Retry Failures")
                print("Maximum retry attempts.")
                raise

def DownloadData(stellar_class, maxIndex = -1):
    MnemonicDict = {
        'O': 'Oh,',
        'B': 'Be',
        'A': 'A',
        'F': 'Fine',
        'G': 'Girl',
        'K': 'Kiss',
        'M': 'Me'
        }
    
    dataset = Table.read(DataListDir + f'Class{stellar_class}_Data_Annex.fits')
    print(MnemonicDict[stellar_class])
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
            print(f'Finished {i}/{len(dataset)}: {samplename}, Class {stellar_class}')
            continue
        data = GetSpectra(samplestar)
        spectrum = data[0]
        spectrum[1].writeto(candidate, overwrite=False)
        print(f'Finished {i}/{len(dataset)}: {samplename}, Class {stellar_class}')
        time.sleep(20)
    
    print(MnemonicDict[stellar_class] + ' completed')

def StoreSpectralData():
    maxIndex = 2000
    stellar_class = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    logging.info(f'Spectral Data to download: {maxIndex} per class')
    
    for s in stellar_class:
        DownloadData(s, maxIndex)
        time.sleep(600)

def SearchClassBounds(stellar_class, fluxMax, lmbMin, lmbMax):
    MnemonicDict = {
        'O': 'Oh,',
        'B': 'Be',
        'A': 'A',
        'F': 'Fine',
        'G': 'Girl',
        'K': 'Kiss',
        'M': 'Me'
        }
    
    dataset = Table.read(DataListDir + f'Class{stellar_class}_Data_Annex.fits')
    print(MnemonicDict[stellar_class])
    print(len(dataset))
    maxIndex = 1500
    
    for i in range(maxIndex):
        samplestar = Table(dataset[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DataFileDir + f'{samplename}.fits') as hdul:
            test = hdul[1].data
        
        fluxMax = max(fluxMax, np.max(test['flux']))
        lmbMin = min(lmbMin, 10**np.min(test['loglam']))
        lmbMax = max(lmbMax, 10**np.max(test['loglam']))
    
    return (fluxMax, lmbMin, lmbMax)

def GetDataBounds():
    fluxMax = 0
    lmbMin = np.inf
    lmbMax = -np.inf
    
    stellar_class = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    for s in stellar_class:
        fluxMax, lmbMin, lmbMax = SearchClassBounds(s, fluxMax, lmbMin, lmbMax)
    
    print(f'fluxMax = {fluxMax}')
    print(f'lmbMin = {lmbMin}')
    print(f'lmbMax = {lmbMax}')
    
    ### Original 1000
    # fluxMax = 3460.751708984375
    # lmbMin = 3553.0396325911483
    # lmbMax = 10387.235932988217
    
    ### Annex 1500
    # fluxMax = 11039.7724609375
    # lmbMin = 3557.132279448977
    # lmbMax = 10403.9916060202

def PrepDataIndexing (table):
    table['subclass'] = table['subclass'].astype(str)
    array = table[['specobjid', 'subclass']].as_array()
    np.random.shuffle(array)
    return Table(array)


def PartitionTrainingData():
    results_o = Table.read(DataListDir + 'ClassO_Data_Annex.fits')[:1500]
    results_b = Table.read(DataListDir + 'ClassB_Data_Annex.fits')[:1500]
    results_a = Table.read(DataListDir + 'ClassA_Data_Annex.fits')[:1500]
    results_f = Table.read(DataListDir + 'ClassF_Data_Annex.fits')[:1500]
    results_g = Table.read(DataListDir + 'ClassG_Data_Annex.fits')[:1500]
    results_k = Table.read(DataListDir + 'ClassK_Data_Annex.fits')[:1500]
    results_m = Table.read(DataListDir + 'ClassM_Data_Annex.fits')[:1500]
    
    results_o = PrepDataIndexing(results_o)
    results_b = PrepDataIndexing(results_b)
    results_a = PrepDataIndexing(results_a)
    results_f = PrepDataIndexing(results_f)
    results_g = PrepDataIndexing(results_g)
    results_k = PrepDataIndexing(results_k)
    results_m = PrepDataIndexing(results_m)
    
    trainingdata = vstack([results_o[['specobjid', 'subclass']][:1200], results_b[['specobjid', 'subclass']][:1200], 
                           results_a[['specobjid', 'subclass']][:1200], results_f[['specobjid', 'subclass']][:1200], 
                           results_g[['specobjid', 'subclass']][:1200], results_k[['specobjid', 'subclass']][:1200], 
                           results_m[['specobjid', 'subclass']][:1200]
                         ])
    trainingdata = PrepDataIndexing(trainingdata)
    
    testingdata = vstack([results_o[['specobjid', 'subclass']][-300:], results_b[['specobjid', 'subclass']][-300:], 
                          results_a[['specobjid', 'subclass']][-300:], results_f[['specobjid', 'subclass']][-300:], 
                          results_g[['specobjid', 'subclass']][-300:], results_k[['specobjid', 'subclass']][-300:], 
                          results_m[['specobjid', 'subclass']][-300:]
                        ])
    testingdata = PrepDataIndexing(testingdata)
    
    trainingdata.write(DataListDir + 'Training_Data_Annex.fits', format='fits', overwrite=True)
    testingdata.write(DataListDir + 'Testing_Data_Annex.fits', format='fits', overwrite=True)

### Disabled ### QueryStars()

### Disabled ### StoreSpectralData()

### Disabled ### GetDataBounds()

### Disabled ### PartitionTrainingData()

print('success')