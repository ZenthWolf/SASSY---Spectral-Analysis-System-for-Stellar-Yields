# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:13:07 2023

@author: Zenth
"""

from astropy.table import Table
from astropy.io import fits

import json
import math
import numpy as np

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.model_selection import GridSearchCV
#from keras.layers import LeakyReLU

import SASSY_Analysis_Tools as SASSY

DATA_LIST_DIR = 'SDSS_Data_Subset/'
DATA_FILE_DIR = DATA_LIST_DIR + 'Spectra/'
PLOTTING_DIR = 'Learning_Plots/'

SPECTRA_MIN_COUNT = 2888
SPECTRA_MAX_COUNT = 4633

# Original Survey
FLUX_MAX = 252.4149169921875
LAMBDA_MIN = 3552.2204983897977
LAMBDA_MAX = 10399.206408133097

WINDOW_SIZE = 5

LOSS_RANGE = [0.0, 1.0]
ACCURACY_RANGE = [0.6, 1.0]

# Metrics
training_loss = []
training_accuracy = []

testing_loss = []
testing_accuracy = []

precision_rating = []
recall_rating = []
f1_score = []

report_history = []

training_precision_rating = []
training_recall_rating = []
training_f1_score = []

training_report_history = []

placing_hist = {
    'B': [],
    'A': [],
    'F': [],
    'G': [],
    'K': [],
    'M': []
}

content_hist  = {
    'B': [],
    'A': [],
    'F': [],
    'G': [],
    'K': [],
    'M': []
}

training_placing_hist = {
    'B': [],
    'A': [],
    'F': [],
    'G': [],
    'K': [],
    'M': []
}

training_content_hist  = {
    'B': [],
    'A': [],
    'F': [],
    'G': [],
    'K': [],
    'M': []
}

# Dataset
trainingTable = Table.read(DATA_LIST_DIR + 'Training_Data_Annex.fits')
testingTable = Table.read(DATA_LIST_DIR + 'Testing_Data_Annex.fits')

class_mapping = {
    'B': np.array([0]),
    'A': np.array([1]),
    'F': np.array([2]),
    'G': np.array([3]),
    'K': np.array([4]),
    'M': np.array([5])
}

def PrepData(sourceTable, localNorm = True):
    input_data_list = []
    output_data_list = []
    
    for i in range(len(sourceTable)):
        if i%100 == 0:
            print(f'{i} / {len(sourceTable)}')
        testingStar = Table(sourceTable[i])
        starName = testingStar['specobjid'][0]
        with fits.open(DATA_FILE_DIR + f'{starName}.fits') as hdu:
            spectrum = hdu[1].data
        
        # Normalize
        fluxMax = FLUX_MAX
        if localNorm:
            flux = np.clip(spectrum['flux'], 0, np.inf)
            fluxMax = np.max(flux)
        flux = spectrum['flux']/fluxMax
        flux = np.clip(flux, 0, np.inf)
        wavelength = (10 ** spectrum['loglam'] - LAMBDA_MIN)/LAMBDA_MAX
        
        # Smooth
        flux = np.convolve(flux, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
        
        if (zeros := SPECTRA_MAX_COUNT - len(flux)) > 0:
            zero_array = np.zeros(zeros)
            flux = np.append(flux, zero_array)
            wavelength = np.append(wavelength, zero_array)
        
        sample_data = np.array([np.concatenate((flux[:, np.newaxis], wavelength[:, np.newaxis]), axis=1)])
        sample_class = class_mapping[testingStar['subclass'][0][0]]
        
        input_data_list.append(sample_data)
        output_data_list.append(sample_class)
    
    input_data = np.array(input_data_list)
    output_data = np.array(output_data_list)
    return (input_data, output_data)

(x_train, y_train) = PrepData(trainingTable, localNorm = True)
(x_test, y_test) = PrepData(testingTable, localNorm = True)

#x_train = np.load('x_train.npy')
#y_train = np.load('y_train.npy')
#x_test = np.load('x_test.npy')
#y_test = np.load('y_test.npy')

def create_model(nodes_per_layer, learning_rate, dropout_rate):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv1D(32, 32, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    
    for nodes in nodes_per_layer:
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def Train_Model(epochs):
    for epoch in range(epochs):
        training = model.fit(x_train, y_train, epochs=1, shuffle=True)
        training_loss.append(training.history['loss'][0])
        training_accuracy.append(training.history['accuracy'][0])
        
        testing = model.evaluate(x_test, y_test, verbose=2)
        testing_loss.append(testing[0])
        testing_accuracy.append(testing[1])
        
        prediction =  model.predict(x_test)
        predicted_labels = np.argmax(prediction, axis=1)
        true_labels = np.squeeze(y_test)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
        precision_rating.append(precision)
        recall_rating.append(recall)
        f1_score.append(f1)
        
        report, placement, containment = SASSY.AnalyzeClassification (predicted_labels, true_labels)
        
        report_history.append(report)
        
        for key in placement:
            placing_hist[key].append(placement[key])
            content_hist[key].append(containment[key])
        
        training_prediction =  model.predict(x_train)
        training_predicted_labels = np.argmax(training_prediction, axis=1)
        training_true_labels = np.squeeze(y_train)
        
        precision, recall, f1, _ = precision_recall_fscore_support(training_true_labels, training_predicted_labels, average=None)
        training_precision_rating.append(precision)
        training_recall_rating.append(recall)
        training_f1_score.append(f1)
        
        report, placement, containment = SASSY.AnalyzeClassification (training_predicted_labels, training_true_labels)
        
        training_report_history.append(report)
        
        for key in placement:
            training_placing_hist[key].append(placement[key])
            training_content_hist[key].append(containment[key])
        
        print(f'\nEpochs Completed: {epoch+1} / {epochs}\n')
    
    return

layers = [700, 700]
initial_learning = 0.0001
dropout_rate = 0.1

model = create_model(layers, initial_learning, dropout_rate)

file_name = 'Layer'

for l in layers:
    file_name += f'-{l}'

file_name += f'_Learning-E{int(int(math.log10(initial_learning)))}'
file_name += f'_Window-{WINDOW_SIZE}'
file_name += '_Conv-32-10'

print("\n!TRAINING!\n")

Train_Model(3)

model.optimizer.learning_rate.assign(initial_learning/10)

Train_Model(5)

model.optimizer.learning_rate.assign(initial_learning/100)

Train_Model(142)

model.optimizer.learning_rate.assign(initial_learning/500)

Train_Model(1850)

SASSY.Plot_Learning(training_loss, training_accuracy, file_name, 'Training', PLOTTING_DIR, isExtended = False)
SASSY.Plot_Learning(testing_loss, testing_accuracy, file_name, 'Testing', PLOTTING_DIR, isExtended = False)

SASSY.Plot_Learning_Comparison(training_loss, testing_loss, file_name, 'Loss', PLOTTING_DIR, LOSS_RANGE)
SASSY.Plot_Learning_Comparison(training_accuracy, testing_accuracy, file_name, 'Accuracy', PLOTTING_DIR, ACCURACY_RANGE)

'''
SASSY.Plot_Category_Metric(precision_rating, file_name, 'Validation_Precision', PLOTTING_DIR, save = False)
SASSY.Plot_Category_Metric(recall_rating, file_name, 'Validation_Recall', PLOTTING_DIR, save = False)
SASSY.Plot_Category_Metric(f1_score, file_name, 'Validation_F1-Score', PLOTTING_DIR, save = False)

SASSY.Plot_Category_Metric(training_precision_rating, file_name, 'Training_Precision', PLOTTING_DIR, save = False)
SASSY.Plot_Category_Metric(training_recall_rating, file_name, 'Training_Recall', PLOTTING_DIR, save = False)
SASSY.Plot_Category_Metric(training_f1_score, file_name, 'Training_F1-Score', PLOTTING_DIR, save = False)

SASSY.Plot_Category_Hist(placing_hist, file_name, "Validation_Placement", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(training_placing_hist, file_name, "Training_Placement", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(content_hist, file_name, "Validation_Contents", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(training_content_hist, file_name, "Training_Contents", PLOTTING_DIR, save = False)
'''

data = np.column_stack((training_accuracy, training_loss))
np.savetxt(PLOTTING_DIR + f'{file_name}_Training.csv', data, delimiter=',')

data = np.column_stack((testing_accuracy, testing_loss))
np.savetxt(PLOTTING_DIR + f'{file_name}_Testing.csv', data, delimiter=',')

def WriteAnalysis(metric, name, needsConversion = False):
    if needsConversion:
        metric = [array.tolist() for array in metric]
    
    file_path = PLOTTING_DIR + file_name + f'_{name}_Validation.json'
    with open(file_path, "w") as file:
        json.dump(metric, file)

def SaveAnalysis():
    WriteAnalysis(precision_rating, 'precision_rating', needsConversion=True)
    WriteAnalysis(training_precision_rating, 'training_precision_rating', needsConversion=True)
    
    WriteAnalysis(recall_rating, 'recall_rating', needsConversion=True)
    WriteAnalysis(training_recall_rating, 'training_recall_rating', needsConversion=True)
    
    WriteAnalysis(f1_score, 'f1_score', needsConversion=True)
    WriteAnalysis(training_f1_score, 'training_f1_score', needsConversion=True)
    
    WriteAnalysis(report_history, 'report_history')
    WriteAnalysis(training_report_history, 'training_report_history')
    
    WriteAnalysis(placing_hist, 'placing_hist')
    WriteAnalysis(training_placing_hist, 'training_placing_hist')
    
    WriteAnalysis(content_hist, 'content_hist')
    WriteAnalysis(training_content_hist, 'training_content_hist')
