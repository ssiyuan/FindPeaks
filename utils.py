""" Utility functions for working with molecule data. """

import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


def read_molecule_data(file_path):
    """ Return the data from file_path, and convert to the right format.
        The input should be the path to a .csv file.
    """
    data = []
    with file_path.open(mode='r', encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",", quotechar='\"')
        for line in csv_reader:
            data.append(line)

    # The data in file is read by column, so transpose it to read by row.
    data_needed = np.array(data).transpose()
    data_needed = data_needed.astype(np.float32)  # Convert string to float.
    return data_needed


def separate_data(data): 
    """ Return 2_theta (x-axis) and sets of data for the molecule (y-axis).

    Input: 2-D array, the first line is for x-axis, other lines for y-axis.
    """
    x = data[0]  
    ys = data[1:] 
    return x, ys


def plot_initial_figure(x, ys):
    """ Draw a figure for the data read from file. 

    Inputs:
        x: 1-D array.
        ys: 2-D array. 
    """
    for i in range(ys.shape[0]):
        plt.plot(x, ys[i], linewidth = 0.6)
    plt.xlabel('2-theta') 
    plt.ylabel('Intensity')
    plt.show() 


def plot_peaks(x, ys):
    """ Draw a figure for the data read from file and mark the peaks. 

    Inputs:
        x: 1-D array.
        ys: 2-D array.  
    """
    for i in range(ys.shape[0]):
    # for i in range(13,14):
        current_y = ys[i]
        plt.plot(x, current_y, linewidth = 0.6)  # the lines
        # Look for the peaks:
        peak_indices, peak_property = find_peaks(current_y, height=0, 
                                    prominence=(0.0063, None))
        plt.plot(x[peak_indices], current_y[peak_indices], ".")
    # plt.plot(x, ys[0], linewidth = 0.6)
    plt.xlabel('2-theta') 
    plt.ylabel('Intensity') 
    
    # plt.plot(x[peak_indices], ys[1][peak_indices], "o")
    plt.show()