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
    """ Return an array of indices, corresponding to the elements in the given 
    range.

    Inputs:
        x: 1-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    return np.where((x>x_range[0]) & (x<x_range[1]))[0]


# 1
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


# 1.5
def find_interval_peak(x, y, x_range):
    """ Return the index of a peak in an interval. 
    (Use function find_peaks() directly.)

    The inputs:
        x: 1-D array.
        y: 1-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    interval_indices = indices_in_interval(x, x_range)
    index_range_min = interval_indices[0]
    y_in_interval = y[interval_indices]
    temp_index, peak_property = find_peaks(y_in_interval, height=0, distance=100)
    peak_index = index_range_min + temp_index
    return peak_index, peak_property


def print_peak_positions(peaks_positions, x_range):
    """ Print positions for peaks in an interval. 
    
    Inputs: 
        peaks_positions: 2-D array of 2 columns
        x_range: 1-D list or array of 2 elements
    """
    print(f"In interval [{x_range[0]}, {x_range[1]}]:")
    for i in range(len(peaks_positions)):
        if len(peaks_positions[i]) == 1:  # element is -1, meaning no value
            print(f"    {i}th set of data has no peak here. ")
        elif len(peaks_positions[i]) == 2:
            print(f"    {i}th set of data has peak at: {peaks_positions[i]}. ")


# 1. look for peaks in an interval (by maximum value)
def plot_maxes_in_range(x, ys, x_range):
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


# 1.5. look for peaks in an interval (by using function find_peaks())
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
        peak_index, peak_property = find_interval_peak(x, current_y, x_range)

        plt.plot(x[peak_index], current_y[peak_index], ".")

        # print(x[peak_index])
        # print(current_y[peak_index])

        current_peak = []
        if len(peak_index) == 0: # no peak
            current_peak.append(-1)
        else:
            current_peak.append(x[peak_index][0])
            current_peak.append(current_y[peak_index][0])
        peaks_positions.append(current_peak)

    plt.xlabel('2-theta') 
    plt.ylabel('Intensity') 
    plt.show()

    print_peak_positions(peaks_positions, x_range)


# 2. look for all peaks 
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
























def indices_in_intervals(x, x_ranges):
    """ Return a 2-D list of indices, corresponding to elements in the given 
    ranges. Each row is for an interval.

    Inputs:
        x: 1-D array.
        x_ranges: 2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range
    """
    indices_in_intervals = []
    for x_range in x_ranges:
        indices_in_intervals.append(np.where((x>x_range[0]) & (x<x_range[1]))[0])
    return indices_in_intervals


def find_peaks_in_ranges(x, y, x_ranges):
    """ Return indices and properties of peaks in one or more given intervals. 
    (Use function find_peaks() directly.)

    The inputs:
        x: 1-D array.
        y: 1-D array.
        x_ranges: 2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range.
    """
    intervals_indices = indices_in_intervals(x, x_ranges)
    peak_indices, peak_properties= [], []
    for interval_indices in intervals_indices: 
        index_range_min = interval_indices[0]
        y_in_interval = y[interval_indices]
        temp_index, peak_property = find_peaks(y_in_interval, height=0, distance=100)
        peak_index = index_range_min + temp_index

        peak_indices.append(peak_index)
        peak_properties.append(peak_property)

    return peak_indices, peak_properties


# 改注释
def print_peaks_positions(peaks_positions, x_ranges):
    """ Print positions for peaks in an interval. 
    
    Inputs: 
        peaks_positions: 3-D array of 2 columns
        x_range: 2-D list or array of 2 elements
    """
    j = 0  # j: the j-th interval
    for x_range in x_ranges:
        print(f"\n\nIn interval [{x_range[0]}, {x_range[1]}]:")
        for i in range(len(peaks_positions)):  # i: the i-th set of data
            if len(peaks_positions[i][j]) == 1:  # element is -1: no value
                print(f"{i}th set of data has no peak here.")
            elif len(peaks_positions[i][j]) == 2:
                print(f"{i}th set of data has peak at: {peaks_positions[i][j]}.")
        j += 1


def list_to_tuple(x_ranges):
    """ If the input is a list of shape (2,), turn it to a tuple. 
    e.g. [1,1] --> [[1,1]]
    """
    temp = []
    temp.append(x_ranges)
    return temp


# 改注释
# 3. look for peaks in several intervals (by using function find_peaks())
def plot_peaks_in_ranges(x, ys, x_ranges):
    """ Draw a figure for the data read from file and mark the peaks for an 
    interval. 
    
    The inputs:
        x: 1-D array.
        ys: 2-D array.
        x_range: 2-D list or array of 2 elements, the minimum and maximum of 
                 the range. 
    """
    if np.array(x_ranges).shape == (2,):  # check the input format 
        x_ranges = list_to_tuple(x_ranges)

    peaks_positions = []
    for i in range(len(ys)):  # the i-th data set for y-axis
        current_y = ys[i]
        plt.plot(x, current_y, linewidth = 0.6)  # the lines
        peak_indices, peak_properties = find_peaks_in_ranges(x, current_y, 
                                        x_ranges)

        current_peaks = []  # peaks for a data set in several intervals
        for j in range(len(x_ranges)):  # the j-th interval on x-axis
            peak_index = peak_indices[j]
            peak_x = x[peak_index]
            peak_y = current_y[peak_index]
            plt.plot(peak_x, peak_y, ".")

            current_peak = []  # the peak for an interval
            if len(peak_index) == 0: # no peak
                current_peak.append(-1)
            else:
                current_peak.append(peak_x[0])
                current_peak.append(peak_y[0])
            current_peaks.append(current_peak)

        peaks_positions.append(current_peaks)

    plt.xlabel('2-theta') 
    plt.ylabel('Intensity') 
    plt.show()

    print_peaks_positions(peaks_positions, x_ranges)
