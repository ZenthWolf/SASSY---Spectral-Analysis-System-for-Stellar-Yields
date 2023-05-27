# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:13:07 2023

@author: Zenth
"""

from astropy.table import Table
from astropy.io import fits

import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.model_selection import GridSearchCV
#from keras.layers import LeakyReLU


DATA_LIST_DIR = 'SDSS_Data_Subset/'
DATA_FILE_DIR = DATA_LIST_DIR + 'Spectra/'
PLOTTING_DIR = 'Learning_Plots/'

SPECTRA_MIN_COUNT = 2888
SPECTRA_MAX_COUNT = 4633

# Original Survey
FLUX_MAX = 252.4149169921875
LAMBDA_MIN = 3552.2204983897977
LAMBDA_MAX = 10399.206408133097
#LAMBDA_MIN = np.log10(LAMBDA_MIN)
#LAMBDA_MAX = np.log10(LAMBDA_MAX)

# From current data set
# FLUX_MAX = 3460.751708984375
# LAMBDA_MIN = 3553.0396325911483
# LAMBDA_MAX = 10387.235932988217

# Training Parameters:

WINDOW_SIZE = 5
EPOCHS = 85

trainingTable = Table.read(DATA_LIST_DIR + 'Training_Data_Annex.fits')
testingTable = Table.read(DATA_LIST_DIR + 'Testing_Data_Annex.fits')

def CheckSpectralRange():
    spectralSizeMax = -np.inf
    spectralSizeMin = np.inf
    
    for i in range(len(trainingTable)):
        if i%100 == 0:
            print(f'{i} / {len(trainingTable)}')
        samplestar = Table(trainingTable[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DATA_FILE_DIR + f'{samplename}.fits') as hdu:
            spectrum = hdu[1].data
        testlen = len(spectrum['flux'])
        spectralSizeMax = max(spectralSizeMax, testlen)
        spectralSizeMin = min(spectralSizeMin, testlen)
        print(spectralSizeMin)
        print(spectralSizeMax)
    
    for i in range(len(testingTable)):
        if i%100 == 0:
            print(f'{i} / {len(testingTable)}')
        samplestar = Table(testingTable[i])
        samplename = samplestar['specobjid'][0]
        with fits.open(DATA_FILE_DIR + f'{samplename}.fits') as hdu:
            spectrum = hdu[1].data
        testlen = len(spectrum['flux'])
        spectralSizeMax = max(spectralSizeMax, testlen)
        spectralSizeMin = min(spectralSizeMin, testlen)
    
    print(spectralSizeMin)
    print(spectralSizeMax)

'''
class_mapping = {
    'O': np.array([0]),
    'B': np.array([1]),
    'A': np.array([2]),
    'F': np.array([3]),
    'G': np.array([4]),
    'K': np.array([5]),
    'M': np.array([6])
}
'''

class_mapping = {
    'B': np.array([0]),
    'A': np.array([1]),
    'F': np.array([2]),
    'G': np.array([3]),
    'K': np.array([4]),
    'M': np.array([5])
}

def PrepData(sourceTable):
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
        flux = spectrum['flux']/FLUX_MAX
        flux = np.clip(flux, 0, np.inf)
        wavelength = (10 ** spectrum['loglam'] - LAMBDA_MIN)/LAMBDA_MAX
        
        # Smooth
        flux = np.convolve(flux, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
        
        #if i==0:
        #    plt.plot(wavelength, flux)
        #    plt.xlabel('Wavelength [$\AA$]')
        #    plt.ylabel('Flux [10$^{-17}$ erg/cm$^2$/s/$\AA$]')
        #    plt.show()
        
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

def PrepDataLocalNorm(sourceTable):
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
        flux = np.clip(spectrum['flux'], 0, np.inf)
        localFluxMax = np.max(flux)
        flux = spectrum['flux']/localFluxMax
        wavelength = (10 ** spectrum['loglam'] - LAMBDA_MIN)/LAMBDA_MAX
        
        # Smooth
        flux = np.convolve(flux, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
        
        #if i==0:
        #    plt.plot(wavelength, flux)
        #    plt.xlabel('Wavelength [$\AA$]')
        #    plt.ylabel('Flux [10$^{-17}$ erg/cm$^2$/s/$\AA$]')
        #    plt.show()
        
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

#(x_train, y_train) = PrepData(trainingTable)
#(x_test, y_test) = PrepData(testingTable)

(x_train, y_train) = PrepDataLocalNorm(trainingTable)
(x_test, y_test) = PrepDataLocalNorm(testingTable)

#x_train = np.load('x_train.npy')
#y_train = np.load('y_train.npy')
#x_test = np.load('x_test.npy')
#y_test = np.load('y_test.npy')

def AnalyzeClassification (predicted_labels, true_labels):
    key = ['B', 'A', 'F', 'G', 'K', 'M']
    
    placement = {
        'B': [0, 0, 0, 0, 0, 0],
        'A': [0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0]
    }
    
    containment  = {
        'B': [0, 0, 0, 0, 0, 0],
        'A': [0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0]
    }
    
    report = classification_report(true_labels, predicted_labels)
    
    for ind, label in enumerate(predicted_labels):
        truth = true_labels[ind]
        
        label_key = key[label]
        truth_key = key[truth]
        
        placement[truth_key][label] += 1
        containment[label_key][truth] += 1
    
    return (report, placement, containment)


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

layers = [700, 700]
initial_learning = 0.0001

model = create_model(layers, initial_learning, 0.1)

file_name = 'Layer'

for l in layers:
    file_name += f'-{l}'

file_name += f'_Learning-E{int(int(math.log10(initial_learning)))}'
file_name += f'_Window-{WINDOW_SIZE}'
file_name += '_Conv-32-10'

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

def Train_Model(epochs = EPOCHS):
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
        
        report, placement, containment = AnalyzeClassification (predicted_labels, true_labels)
        
        report_history.append(report)
        
        for key in placement:
            placing_hist[key].append(placement[key])
            content_hist[key].append(content_hist[key])
        
        training_prediction =  model.predict(x_train)
        training_predicted_labels = np.argmax(training_prediction, axis=1)
        training_true_labels = np.squeeze(y_train)
        
        report, placement, containment = AnalyzeClassification (training_predicted_labels, training_true_labels)
        
        training_report_history.append(report)
        
        for key in placement:
            training_placing_hist[key].append(placement[key])
            training_content_hist[key].append(content_hist[key])
        
        print(f'\nEpochs Completed: {epoch+1} / {epochs}\n')
    
    return

def Print_Learning(isExtended = False):
    plot_suffix = '.png'
    data_suffix = '.csv'
    if isExtended:
        plot_suffix = '_Long' + plot_suffix
        data_suffix = '_Long' + data_suffix
    
    plt.figure()
    plt.plot(range(len(training_accuracy)), training_accuracy, label='Train Accuracy')
    plt.plot(range(len(training_loss)), training_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Accuracy and Loss')
    plt.legend()
    
    plt.savefig(PLOTTING_DIR + f'{file_name}_Training' + plot_suffix)
    
    plt.show()
    data = np.column_stack((training_accuracy, training_loss))
    np.savetxt(PLOTTING_DIR + f'{file_name}_Training'  + data_suffix, data, delimiter=',')
    
    plt.figure()
    plt.plot(range(len(testing_accuracy)), testing_accuracy, label='Validation Accuracy')
    plt.plot(range(len(testing_loss)), testing_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Accuracy and Loss')
    plt.legend()
    
    plt.savefig(PLOTTING_DIR + f'{file_name}_Validation' + plot_suffix)
    
    plt.show()
    data = np.column_stack((testing_accuracy, testing_loss))
    np.savetxt(PLOTTING_DIR + f'{file_name}_Validation'  + data_suffix, data, delimiter=',')

def ExtractReportMetrics(metric):
    star_metrics = {
        'B': [],
        'A': [],
        'F': [],
        'G': [],
        'K': [],
        'M': []
    }
    
    for m in metric:
        for i, star_type in enumerate(star_metrics.keys()):
            star_metrics[star_type].append(m[i])
        
    return star_metrics

def Compare (LossRange = [0.3, 0.8], AccRange = [0.6, 1.0], save = False):
    plt.figure()
    plt.plot(range(len(training_loss)), training_loss, label='Training')
    plt.plot(range(len(testing_loss)), testing_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.ylim(LossRange)
    plt.legend()
    
    if save:
        plt.savefig(PLOTTING_DIR + f'{file_name}_Loss_Comparison.png')
    
    plt.show()
    
    plt.figure()
    plt.plot(range(len(training_accuracy)), training_accuracy, label='Training')
    plt.plot(range(len(testing_accuracy)), testing_accuracy, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(AccRange)
    plt.legend()
    
    if save:
        plt.savefig(PLOTTING_DIR + f'{file_name}_Accuracy_Comparison.png')
    
    plt.show()
    
    star_metrics = ExtractReportMetrics(precision_rating)
    
    plt.figure()
    for star_type in star_metrics:
        data = star_metrics[star_type]
        plt.plot(range(len(data)), data, label=f'{star_type}-Type')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision by Category')
    plt.ylim(AccRange)
    plt.legend()
    
    if save:
        plt.savefig(PLOTTING_DIR + f'{file_name}_Category_Precision.png')
    
    plt.show()
    
    star_metrics = ExtractReportMetrics(recall_rating)
    
    plt.figure()
    for star_type in star_metrics:
        data = star_metrics[star_type]
        plt.plot(range(len(data)), data, label=f'{star_type}-Type')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall by Category')
    plt.ylim(AccRange)
    plt.legend()
    
    if save:
        plt.savefig(PLOTTING_DIR + f'{file_name}_Category_Recall.png')
    
    plt.show()
    
    star_metrics = ExtractReportMetrics(f1_score)
    
    plt.figure()
    for star_type in star_metrics:
        data = star_metrics[star_type]
        plt.plot(range(len(data)), data, label=f'{star_type}-Type')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-Score by Category')
    plt.ylim(AccRange)
    plt.legend()
    
    if save:
        plt.savefig(PLOTTING_DIR + f'{file_name}_Category_F1-Score.png')
    
    plt.show()


#ParamList = []
#ly = [700, 700]
#il = int(int(math.log10(0.0001)))
#WS = 1
#ParamList.append([ly, il, WS])

def CombinePlot(ParamList):
    plt.figure()
    
    for name in ParamList:
        file_name = name[3] + 'Layer'
        label_name= 'Lys'
        
        for l in name[0]:
            file_name += f'-{l}'
            label_name += f'-{l}'
            
        
        file_name += f'_Learning-E{name[1]}'
        label_name += f'_Lr-E{name[1]}'
        
        file_name += f'_Window-{name[2]}'
        label_name += f'_WS-{name[2]}'
        
        file_name += '_Conv-32-10_Validation.csv'
        
        label_name = name[3]
        if label_name == '':
            label_name = 'Base_'
        
        data = np.loadtxt(PLOTTING_DIR + file_name, delimiter=',')
        accuracy = []
        for p in data:
            accuracy.append(p[0])
        plt.plot(range(len(accuracy)), accuracy, label=label_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim([0.6, 1.0])
    plt.legend()
    
    plt.savefig(PLOTTING_DIR + 'Combined_Comparison_Validation.png')
    
    plt.show()


print("\n!TRAINING!\n")

Train_Model(3)

model.optimizer.learning_rate.assign(initial_learning/10)

Train_Model(5)

model.optimizer.learning_rate.assign(initial_learning/100)

Train_Model(492)

Print_Learning()

Compare()

# model.save('model.h5')
# loaded_model = tf.keras.models.load_model('model.h5')
# OR
# model.load_weights('my_model_weights.h5')


#Reporting
# from sklearn.metrics import classification_report
#y_pred = model.predict(x_test)

#y_pred_labels = np.argmax(y_pred, axis=1)

#y_true_labels = np.squeeze(y_test)

#report = classification_report(y_true_labels, y_pred_labels)
#print(report)

