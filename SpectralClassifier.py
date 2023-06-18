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

#from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

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

FRONT_LAYER = 0
LAYERS = [128, 128, 128]
BASE_LEARNING = 0.0001
INITIAL_LEARNING = BASE_LEARNING
DROPOUT_RATE = 0.1
WINDOW_SIZE = 10
NUMBER_FILTERS = 32
KERNEL_SIZE = 32

FILE_NAME = 'Multilayer_128_128_128_SmoothGauss'
EXTEND_NAME = False

if EXTEND_NAME:
    for l in LAYERS:
        FILE_NAME += f'-{l}'
    
    FILE_NAME += f'_Learning-E{int(int(math.log10(INITIAL_LEARNING)))}'
    FILE_NAME += f'_Window-{WINDOW_SIZE}'
    FILE_NAME += f'_Conv-{NUMBER_FILTERS}-{KERNEL_SIZE}'

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

with open(DATA_LIST_DIR + 'subclasses.json', "r") as file:
    subclasses = json.load(file)

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
        
        
        flux = spectrum['flux']
        
        # Smooth
        if WINDOW_SIZE > 1:
            flux = np.convolve(flux, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
            flux = gaussian_filter1d(flux, 12)
        
        
        # Normalize
        fluxMax = FLUX_MAX
        if localNorm:
        #    flux = np.clip(spectrum['flux'], 0, np.inf)
            fluxMax = np.max(flux)
        flux = flux/fluxMax
        flux = np.clip(flux, 0, np.inf)
        wavelength = (10 ** spectrum['loglam'] - LAMBDA_MIN)/LAMBDA_MAX
        
        
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

def FullClassInformation(sourceTable):
    output_data_list = []
    
    for i in range(len(sourceTable)):
        testingStar = Table(sourceTable[i])
        starID = testingStar['subclass'][0]
        
        output_data_list.append(starID)
    
    return output_data_list

def create_model(nodes_per_layer, learning_rate, dropout_rate):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv1D(NUMBER_FILTERS, KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    
    if FRONT_LAYER > 0:
        model.add(tf.keras.layers.Dense(FRONT_LAYER, activation='relu'))
    
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
        
        UpdateRecollection(predicted_labels, true_labels)
        
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


def WriteAnalysis(metric, name, needsConversion = False):
    if needsConversion:
        metric = [array.tolist() for array in metric]
    
    file_path = PLOTTING_DIR + FILE_NAME + f'_{name}.json'
    with open(file_path, "w") as file:
        json.dump(metric, file)

def ReadAnalysis(name, needsConversion = False):
    file_path = PLOTTING_DIR + FILE_NAME + f'_{name}.json'
    with open(file_path, "r") as file:
        metric = json.load(file)
    
    if needsConversion:
        metric = [np.array(lst) for lst in metric]
    
    return metric

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

def SaveModel(model, filename):
    model.save(f'{filename}.h5')

def SaveWeights(model, filename):
    model.save_weights(f'{filename}.h5')

def LoadModel(filename):
    loaded_model = tf.keras.models.load_model(f'{filename}.h5')
    return loaded_model

def LoadWeights(model, filename):
    model.load_weights(f'{filename}.h5')

def Subclass_Recall_Report(pred_label, true_label, full_label):
    Subclass_Recall = {}
    
    for pred_ind, prediction in enumerate(pred_label):
        if (subclass := full_label[pred_ind]) not in Subclass_Recall:
            Subclass_Recall[subclass] = [0, 0, 0]
        if prediction == true_label[pred_ind]:
            Subclass_Recall[full_label[pred_ind]][0] += 1
        else:
            Subclass_Recall[full_label[pred_ind]][1] += 1
    
    for key in Subclass_Recall:
        Subclass_Recall[key][2] = 100 * Subclass_Recall[key][0] / (Subclass_Recall[key][0] + Subclass_Recall[key][1])
    return Subclass_Recall

def Generate_Testing_Recall_Report():
    pred = model.predict(x_test)
    pred_label = np.argmax(pred, axis=1)

    true_label = y_test.squeeze()
    full_label = FullClassInformation(testingTable)
    
    return Subclass_Recall_Report(pred_label, true_label, full_label)

def Generate_Training_Recall_Report():
    pred = model.predict(x_train)
    pred_label = np.argmax(pred, axis=1)

    true_label = y_train.squeeze()
    full_label = FullClassInformation(trainingTable)
    
    return Subclass_Recall_Report(pred_label, true_label, full_label)

def CollateReports():
    TrainingReport = Generate_Training_Recall_Report()
    TestingReport = Generate_Testing_Recall_Report()
    
    report_lines = []
    for key in subclasses:
        paddedkey = f"{key:20}"

        TrainCount = str(TrainingReport[key][0] + TrainingReport[key][1])
        TrainCount = f'{TrainCount:4}'

        TrainAcc = f'{TrainingReport[key][2]:.2f}'
        TrainAcc = f'{TrainAcc:6}'

        if key in TestingReport:
            TestCount = str(TestingReport[key][0] + TestingReport[key][1])
            TestCount = f'{TestCount:4}'

            TestAcc = f'{TestingReport[key][2]:.2f}'
            TestAcc = f'{TestAcc:6}'

            report_lines.append(f'{paddedkey}: {TrainCount}, {TrainAcc}% || {TestCount}, {TestAcc}%')
        else:
            report_lines.append(f'{paddedkey}: {TrainCount}, {TrainAcc}% || na')
        
    report = "\n".join(report_lines)
    return report, TrainingReport, TestingReport

recall_count = []
for label in FullClassInformation(testingTable):
    recall_count.append([0, label])

recall_count.append([0, 'Epochs'])

def UpdateRecollection(predicted_labels, true_labels):
    for index, label in enumerate(predicted_labels):
        if label == true_labels[index]:
            recall_count[index][0] += 1
    recall_count[len(recall_count)-1][0] += 1

### EXECUTION

with open(PLOTTING_DIR + f'{FILE_NAME}.txt', 'w') as f:
    f.write(f'Layers: {FRONT_LAYER}')
    for l in LAYERS:
        f.write(f' -> {l}')
    f.write(' -> 6\n')
    f.write(f'Initial Learning Rate: {INITIAL_LEARNING}\n')
    f.write(f'Dropout Rate: {DROPOUT_RATE}\n')
    f.write(f'Smoothing Window Size: {WINDOW_SIZE}\n')
    f.write(f'Number of filters: {NUMBER_FILTERS}\n')
    f.write(f'Kernel Size: {KERNEL_SIZE}\n')
    
(x_train, y_train) = PrepData(trainingTable, localNorm = True)
(x_test, y_test) = PrepData(testingTable, localNorm = True)


model = create_model(LAYERS, INITIAL_LEARNING, DROPOUT_RATE)

print("\n!TRAINING!\n")

Train_Model(5)

model.optimizer.learning_rate.assign(INITIAL_LEARNING/10)

Train_Model(95)

SASSY.Plot_Learning(training_loss, training_accuracy, FILE_NAME, 'Training', PLOTTING_DIR, isExtended = True)
SASSY.Plot_Learning(testing_loss, testing_accuracy, FILE_NAME, 'Testing', PLOTTING_DIR, isExtended = True)

SASSY.Plot_Learning_Comparison(training_loss, testing_loss, FILE_NAME, 'Loss', PLOTTING_DIR, LOSS_RANGE)
SASSY.Plot_Learning_Comparison(training_accuracy, testing_accuracy, FILE_NAME, 'Accuracy', PLOTTING_DIR, ACCURACY_RANGE)


SASSY.Plot_Category_Metric(precision_rating, FILE_NAME, 'Validation_Precision', PLOTTING_DIR, save = True)
SASSY.Plot_Category_Metric(recall_rating, FILE_NAME, 'Validation_Recall', PLOTTING_DIR, save = True)
SASSY.Plot_Category_Metric(f1_score, FILE_NAME, 'Validation_F1-Score', PLOTTING_DIR, save = True)

SASSY.Plot_Category_Metric(training_precision_rating, FILE_NAME, 'Training_Precision', PLOTTING_DIR, save = True)
SASSY.Plot_Category_Metric(training_recall_rating, FILE_NAME, 'Training_Recall', PLOTTING_DIR, save = True)
SASSY.Plot_Category_Metric(training_f1_score, FILE_NAME, 'Training_F1-Score', PLOTTING_DIR, save = True)

'''
SASSY.Plot_Category_Hist(placing_hist, FILE_NAME, "Validation_Placement", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(training_placing_hist, FILE_NAME, "Training_Placement", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(content_hist, FILE_NAME, "Validation_Contents", PLOTTING_DIR, save = False)
SASSY.Plot_Category_Hist(training_content_hist, FILE_NAME, "Training_Contents", PLOTTING_DIR, save = False)
'''

data = np.column_stack((training_accuracy, training_loss))
np.savetxt(PLOTTING_DIR + f'{FILE_NAME}_Training.csv', data, delimiter=',')

data = np.column_stack((testing_accuracy, testing_loss))
np.savetxt(PLOTTING_DIR + f'{FILE_NAME}_Testing.csv', data, delimiter=',')

SaveAnalysis()

Recall_Report, _, _ = CollateReports()
WriteAnalysis(Recall_Report, 'Recall Report')
WriteAnalysis(recall_count, 'recall_count')