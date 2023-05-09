# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:30:00 2023

@author: Zenth
"""

import time

from astroquery.sdss import SDSS
from astropy.table import Table, vstack
from astropy.io import fits

import numpy as np

import os
print(os.getcwd())

DataListDir = 'SDSS_Data_Subset/'
DataFileDir = DataListDir + 'Spectra/'

def QueryStars():
    # Perform the query and retrieve a random sample of spectra and their corresponding classes
    querybase = "SELECT TOP 1000 s.specobjid, s.mjd, s.fiberID, s.class, s.subclass, s.run2d, s.plate FROM specobj s WHERE s.class='star' AND s.subclass LIKE "
    
    query = querybase + "'O%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassO_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'B%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassB_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'A%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassA_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'F%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassF_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'G%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassG_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'K%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassK_Data.fits', format='fits', overwrite=True)
    
    time.sleep(5)
    
    query = querybase + "'M%'"
    
    results = SDSS.query_sql(query, spectro=True, refresh_query=True, cache=False)
    results.write(DataListDir + 'ClassM_Data.fits', format='fits', overwrite=True)


# Time padding required to prevent overloading servers
# Still seems to be too frequent for the number of requests
# Consider 30 s between each (and toning down between classes) if running again

def StoreSpectralData():
    results_o = Table.read(DataListDir + 'ClassO_Data.fits')
    results_b = Table.read(DataListDir + 'ClassB_Data.fits')
    results_a = Table.read(DataListDir + 'ClassA_Data.fits')
    results_f = Table.read(DataListDir + 'ClassF_Data.fits')
    results_g = Table.read(DataListDir + 'ClassG_Data.fits')
    results_k = Table.read(DataListDir + 'ClassK_Data.fits')
    results_m = Table.read(DataListDir + 'ClassM_Data.fits')
    
    print('Oh,')
    dataset = results_o
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('Be')
    dataset = results_b
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('A')
    dataset = results_a
    print(len(dataset))
    for i in range(len(dataset)):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('Fine')
    dataset = results_f
    print(len(dataset))
    for i in range(len(dataset)):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('Girl')
    dataset = results_g
    print(len(dataset))
    for i in range(len(dataset)):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('Kiss')
    dataset = results_k
    print(len(dataset))
    for i in range(len(dataset)):
        print(f'Entry: {i}')
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)
    
    time.sleep(900)
    
    print('Me')
    dataset = results_m
    print(len(dataset))
    for i in range(len(dataset)):
        print(f'Entry: {i}')
        if i<535:
            continue
        print(dataset[i])
        samplestar = Table(dataset[i])
        data = SDSS.get_spectra(matches=samplestar)
        spectrum = data[0]
        samplename = samplestar['specobjid'][0]
        spectrum[1].writeto(DataFileDir + f'{samplename}.fits', overwrite=True)
        time.sleep(20)

def GetDataBounds():
    results_o = Table.read(DataListDir + 'ClassO_Data.fits')
    results_b = Table.read(DataListDir + 'ClassB_Data.fits')
    results_a = Table.read(DataListDir + 'ClassA_Data.fits')
    results_f = Table.read(DataListDir + 'ClassF_Data.fits')
    results_g = Table.read(DataListDir + 'ClassG_Data.fits')
    results_k = Table.read(DataListDir + 'ClassK_Data.fits')
    results_m = Table.read(DataListDir + 'ClassM_Data.fits')
    
    fluxMax = 0
    lmbMin = 10000
    lmbMax = 0
    
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

PartitionTrainingData()

print('success')