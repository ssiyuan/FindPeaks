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


def indices_in_interval(x, x_range):
    """ Return an array of indices of the input array's elements, which are in 
    the given range.

    Inputs:
        x: 1-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    return np.where((x>x_range[0]) & (x<x_range[1]))[0]


def find_interval_max(x, y, x_range):
    """ Return the index of the maximum value (peak) in an interval. 

    The inputs:
        x: 1-D array.
        y: 1-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    interval_indices = indices_in_interval(x, x_range)
    index_range_min = interval_indices[0]
    y_in_interval = y[interval_indices]
    temp_index = np.where(y_in_interval == np.max(y_in_interval))[0]
    peak_index = index_range_min + temp_index
    return peak_index


def print_peak_positions(peaks_positions, x_range):
    """ Print positions for peaks in an interval. 
    
    Inputs: 
        peaks_positions: 2-D array of 2 columns
        x_range: 1-D list or array of 2 elements
    """
    print(f"In interval [{x_range[0]}, {x_range[1]}]， peaks occur for:")
    for i in range(len(peaks_positions)):
        print(f"    {peaks_positions[i]}")


# 1. 找interval内的peaks
def plot_peaks_in_range(x, ys, x_range):
    """ Draw a figure for the data read from file and mark the peaks for an 
    interval. 
    
    The inputs:
        x: 1-D array.
        ys: 2-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    peaks_positions = []
    for i in range(ys.shape[0]):
        current_y = ys[i]
        plt.plot(x, current_y, linewidth = 0.6)  # the lines
        # Look for the peaks:
        peak_index = find_interval_max(x, current_y, x_range)
        # peak_index, peak_property = find_interval_peak(x, current_y, x_range)

        plt.plot(x[peak_index], current_y[peak_index], ".")

        # print(x[peak_index])
        # print(current_y[peak_index])
        current_peak_position = []
        current_peak_position.append(x[peak_index][0])
        current_peak_position.append(current_y[peak_index][0])  
        peaks_positions.append(current_peak_position)

    plt.xlabel('2-theta') 
    plt.ylabel('Intensity') 
    plt.show()

    print_peak_positions(peaks_positions, x_range)


# 2. 找所有的peaks
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