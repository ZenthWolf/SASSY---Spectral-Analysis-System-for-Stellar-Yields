# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:47:26 2023

@author: Zenth
"""
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report

#####
# Analysis Tools

def AnalyzeClassification (predicted_labels, true_labels):
    """ Identify distribution of how stars are labeled by true type, 
    as well as contents per label for SASSY.
    
    Parameters
    ----------
    predicted_labels : numpy.ndarray
        Labels given by SASSY.
    true_labels : numpy.ndarray
        True labels of data.
    
    Returns
    -------
    report : str
        SKlearn classification_report for this data.
    placement : dict (str -> list(int))
        Histogram of star labels by true type.
    containment : dict (str -> list(int))
        Histogram of true types by SASSY label.
    """
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

#####
# Plotting Tools

def Plot_Learning(loss, accuracy, filename, title, directory, isExtended = False):
    """Plot learning curve and save to disk.
    
    Parameters
    ----------
    loss : list
        Loss calculated each epoch.
    accuracy : list
        Accuracy calculated each epoch.
    filename : str
        Filename- typically reflecting the model.
    title : str
        Title of metrics (e.g 'Training' or 'Validation').
    directory : str
        Relative path to where to save plots.
    isExtended : bool, optional
        Whether this should be considered an extension (in order to preserve original files). The default is False.
    
    Returns
    -------
    None.
    """
    
    epochs = range(len(loss))
    
    plot_suffix = '.png'
    if isExtended:
        plot_suffix = '_Long' + plot_suffix
    
    plt.figure()
    plt.plot(epochs, accuracy, label=f'{title} Accuracy')
    plt.plot(epochs, loss, label=f'{title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f'{title} Learning')
    plt.legend()
    
    plt.savefig(directory + f'{filename}_{title}' + plot_suffix)
    
    plt.show()

def Plot_Learning_Comparison(training, testing, filename, metric, directory, yrange):
    """Plot comparison of a metric in training and testing results and save to disk.
    
    Parameters
    ----------
    training : list
        Metric of training data tracked across multiple epochs.
    testing : list
        Metric of testing data tracked across multiple epochs.
    filename : str
        Filename- typically reflecting the model.
    metric : str
        Name of metric (e.g 'Loss' or 'Accuracy').
    directory : str
        Relative path to where to save plots.
    yrange : list
        Bounds for y-axis display.

    Returns
    -------
    None.

    """
    epochs = range(len(training))
    
    plt.figure()
    plt.plot(epochs, training, label='Training')
    plt.plot(epochs, testing, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} Comparison')
    plt.ylim(yrange)
    plt.legend()
    
    plt.savefig(directory + f'{filename}_{metric}_Comparison.png')
    
    plt.show()

def ExtractReportMetrics(metric):
    """Sorts data of the given metric by stellar class, tracked across epochs.
    
    Parameters
    ----------
    metric : dict (str -> list(int))
        Contains list of the metric per epoch characterizing the data by class. See AnalyzeClassification().

    Returns
    -------
    star_metrics : dict (str -> list(int))
        Contains lists tracking the metric for that label, tracked across all epochs.

    """
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

def Plot_Category_Metric(metric, filename, title, directory, save = False):
    """Plots a metric over the course of all training epochs
    
    Parameters
    ----------
    metric : dict (str -> list(int))
        Contains list of the metric per epoch characterizing the data by class. See AnalyzeClassification().
    filename : str
        Filename- typically reflecting the model.
    title : str
        Title of metrics (e.g 'Training' or 'Validation').
    directory : str
        Relative path to where to save plots.
    save : bool, optional
        If the plot generate will be saved. The default is False.

    Returns
    -------
    None.

    """
    star_metrics = ExtractReportMetrics(metric)
    
    plt.figure()
    for star_type in star_metrics:
        data = star_metrics[star_type]
        plt.plot(range(len(data)), data, label=f'{star_type}-Type')
    plt.xlabel('Epoch')
    plt.ylabel(f'{title}')
    plt.title(f'{title} by Category')
    plt.legend()
    
    if save:
        plt.savefig(directory + f'{filename}_Category_{title}.png')
    
    plt.show()

def Plot_Category_Hist(star_metrics, name, title, directory, save = False):
    # for the placement/contents plots
    for star_type in star_metrics:
        plt.figure()
        data = star_metrics[star_type]
        epochs = range(len(data))

        data_list = {
            'B': [],
            'A': [],
            'F': [],
            'G': [],
            'K': [],
            'M': []
        }

        for epoch in data:
            data_list['B'].append(epoch[0])
            data_list['A'].append(epoch[1])
            data_list['F'].append(epoch[2])
            data_list['G'].append(epoch[3])
            data_list['K'].append(epoch[4])
            data_list['M'].append(epoch[5])

        plt.plot(epochs, data_list['B'], label='B-type')
        plt.plot(epochs, data_list['A'], label='A-type')
        plt.plot(epochs, data_list['F'], label='F-type')
        plt.plot(epochs, data_list['G'], label='G-type')
        plt.plot(epochs, data_list['K'], label='K-type')
        plt.plot(epochs, data_list['M'], label='M-type')

        plt.xlabel('Epoch')
        plt.ylabel('No. Stars')
        plt.title(f'{title} of {star_type}-type Stars')
        plt.legend()

        if save:
            plt.savefig(directory + f'{name}_Category_{title}-{star_type}-Type.png')

        plt.show()

def Plot_Category_Hist_Compare(star_metrics, training_star_metrics, name, title, directory):
    # for the placement/contents plots
    for star_type in star_metrics:
        plt.figure()
        data = star_metrics[star_type]
        training_data = training_star_metrics[star_type]
        epochs = range(len(data))
        
        data_list = {
            'B': [],
            'A': [],
            'F': [],
            'G': [],
            'K': [],
            'M': []
        }
        
        training_data_list = {
            'B': [],
            'A': [],
            'F': [],
            'G': [],
            'K': [],
            'M': []
        }
        
        for epoch in data:
            data_list['B'].append(epoch[0])
            data_list['A'].append(epoch[1])
            data_list['F'].append(epoch[2])
            data_list['G'].append(epoch[3])
            data_list['K'].append(epoch[4])
            data_list['M'].append(epoch[5])
        for epoch in training_data:
            training_data_list['B'].append(epoch[0])
            training_data_list['A'].append(epoch[1])
            training_data_list['F'].append(epoch[2])
            training_data_list['G'].append(epoch[3])
            training_data_list['K'].append(epoch[4])
            training_data_list['M'].append(epoch[5])
        
        plt.plot(epochs, data_list['B'], label='B-type Validation')
        plt.plot(epochs, data_list['A'], label='A-type Validation')
        plt.plot(epochs, data_list['F'], label='F-type Validation')
        plt.plot(epochs, data_list['G'], label='G-type Validation')
        plt.plot(epochs, data_list['K'], label='K-type Validation')
        plt.plot(epochs, data_list['M'], label='M-type Validation')
        
        plt.plot(epochs, training_data_list['B'], label='B-type Training')
        plt.plot(epochs, training_data_list['A'], label='A-type Training')
        plt.plot(epochs, training_data_list['F'], label='F-type Training')
        plt.plot(epochs, training_data_list['G'], label='G-type Training')
        plt.plot(epochs, training_data_list['K'], label='K-type Training')
        plt.plot(epochs, training_data_list['M'], label='M-type Training')
        
        plt.xlabel('Epoch')
        plt.ylabel('No. Stars')
        plt.title(f'{name} {title} of {star_type}-type Stars')
        plt.legend()

        plt.show()

#ParamList = []
#ly = [700, 700]
#il = int(int(math.log10(0.0001)))
#WS = 1
#ParamList.append([ly, il, WS])


def Learning_Cross_Comparison(labels, directory):
    accuracy_data = {}
    loss_data = {}
    
    for label in labels:
        data = np.loadtxt(directory + label + '_Testing.csv', delimiter=',')
        accuracy = data[:,0]
        loss = data[:,1]
        
        accuracy_data[label] = accuracy
        loss_data[label] = loss
    
    plt.figure()
    
    for label in labels:
        epochs = range(len(accuracy_data[label]))
        plt.plot(epochs, accuracy_data[label], label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim([0.6,1.0])
    plt.legend()
    
    plt.show()
    
    plt.figure()
    
    for label in labels:
        epochs = range(len(loss_data[label]))
        plt.plot(epochs, loss_data[label], label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.ylim([0.0,1.0])
    plt.legend()
    
    plt.show()

def Category_Cross_Comparison(labels, directory):
    precision_data = {
        'B': {},
        'A': {},
        'F': {},
        'G': {},
        'K': {},
        'M': {}
    }
    recall_data = {
        'B': {},
        'A': {},
        'F': {},
        'G': {},
        'K': {},
        'M': {}
    }
    f1_data = {
        'B': {},
        'A': {},
        'F': {},
        'G': {},
        'K': {},
        'M': {}
    }
    
    for label in labels:
        with open(directory + label + '_precision_rating.json') as file:
            precision_rating = json.load(file)
        
        precision_rating = [np.array(lst) for lst in precision_rating]
        precision_rating = ExtractReportMetrics(precision_rating)
        
        for star_type in precision_rating:
            precision_data[star_type][label] = precision_rating[star_type]
        
        with open(directory + label + '_recall_rating.json') as file:
            recall_rating = json.load(file)
        
        recall_rating = [np.array(lst) for lst in recall_rating]
        recall_rating = ExtractReportMetrics(recall_rating)
        
        for star_type in recall_rating:
            recall_data[star_type][label] = recall_rating[star_type]
        
        with open(directory + label + '_f1_score.json') as file:
            f1_score = json.load(file)
        
        f1_score = [np.array(lst) for lst in f1_score]
        f1_score = ExtractReportMetrics(f1_score)
        
        for star_type in f1_score:
            f1_data[star_type][label] = f1_score[star_type]
    
    for star_type in precision_data:
        plt.figure()
        for label in labels:
            epochs = range(len(precision_data[star_type][label]))
            plt.plot(epochs, precision_data[star_type][label], label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title(f'Precision Comparison of {star_type} Stars')
        plt.ylim([0.6,1.0])
        plt.legend()
        
        plt.show()
        
    for star_type in recall_data:
        plt.figure()
        for label in labels:
            epochs = range(len(recall_data[star_type][label]))
            plt.plot(epochs, recall_data[star_type][label], label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title(f'Recall Comparison of {star_type} Stars')
        plt.ylim([0.6,1.0])
        plt.legend()
        
        plt.show()
        
    for star_type in f1_data:
        plt.figure()
        for label in labels:
            epochs = range(len(f1_data[star_type][label]))
            plt.plot(epochs, f1_data[star_type][label], label=label)
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title(f'F1-Score Comparison of {star_type} Stars')
        plt.ylim([0.6,1.0])
        plt.legend()
        
        plt.show()
