""" Utility functions for working with molecule data. """

import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit


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
    """ Seperate the first column of the file and other columns.
        1st column: 2-theta (x-axis)
        other columns: intensity (y-axis)
        Return the resulted 2-theta and intensities after log10 processing. 

    Input: 2-D array, the first line is for x-axis, other lines for y-axis.
    """
    x = data[0]  # 2-theta
    ys = data[1:]  # intensity
    # process data with log10 to see more obvious changes in peaks 
    # return np.log10(x), np.log10(ys)
    return x, ys


def plot_initial_figure(x, ys):
    """ Draw a figure for the data read from file. 

    Inputs:
        x: 1-D array.
        ys: 2-D array. 
    """
    for i in range(ys.shape[0]):
        plt.plot(x, ys[i], linewidth = 0.6)
        # plt.loglog(x, ys[i], linewidth = 0.6)
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
    temp_index, peak_property = find_peaks(y_in_interval, height=0.01, distance=100)
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
        plt.loglog(x, current_y, linewidth = 0.6)  # the lines
        # Look for the peaks:
        peak_index = find_interval_max(x, current_y, x_range)
        # peak_index, peak_property = find_interval_peak(x, current_y, x_range)

        plt.loglog(x[peak_index], current_y[peak_index], ".")

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
        plt.loglog(x, current_y, linewidth = 0.6)  # the lines
        # Look for the peaks:
        peak_index, _ = find_interval_peak(x, current_y, x_range)

        plt.loglog(x[peak_index], current_y[peak_index], ".")

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
        plt.loglog(x, current_y, linewidth = 0.6)  # the lines
        # Look for the peaks:
        peak_indices, peak_property = find_peaks(current_y, height=0, 
                                    prominence=(0.0063, None))
        plt.loglog(x[peak_indices], current_y[peak_indices], ".")
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


# !!! to delete: 
def find_peaks_in_ranges(x, y, x_ranges):
    """ Return indices and properties of peaks in one or more given intervals. 
    (Use function find_peaks() directly.)

    The inputs:
        x: 1-D array.
        y: 1-D array.
        x_ranges: 2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range
    """
    intervals_indices = indices_in_intervals(x, x_ranges)
    peak_indices, peak_properties= [], []
    for interval_indices in intervals_indices: 
        index_range_min = interval_indices[0]
        y_in_interval = y[interval_indices]
        temp_index, peak_property = find_peaks(y_in_interval, distance=100)
        # temp: the index of peak in array in given range (x_range)
        peak_index = index_range_min + temp_index 
        # peak_index: index of peak in the whole array x

        peak_indices.append(peak_index)
        peak_properties.append(peak_property)
    return peak_indices, peak_properties


# # 改注释
# def print_peaks_positions(peaks_positions, x_ranges):
#     """ Print positions for peaks in an interval. 
    
#     Inputs: 
#         peaks_positions: 3-D array of 2 columns
#         2-D list or array of 2 columns, 1st col: the minimum of range
#                                                 2nd col: maximum of range
#     """
#     j = 0  # j: the j-th interval
#     for x_range in x_ranges:
#         print(f"\nIn interval [{x_range[0]}, {x_range[1]}]:")
#         for i in range(len(peaks_positions)):  # i: the i-th set of data
#             if len(peaks_positions[i][j]) == 1:  # element is -1: no value
#                 print(f"{i}th set of data has no peak here.")
#             elif len(peaks_positions[i][j]) == 2:
#                 print(f"{i}th set of data has peak at: {peaks_positions[i][j]}.")
#         j += 1


# 改注释
def print_peaks_positions(peaks_positions, x_ranges):
    """ Print positions for peaks in an interval. 
    
    Inputs: 
        peaks_positions: 3-D array of 2 columns
        2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range
    """
    j = 0  # j: the j-th interval
    for x_range in x_ranges:
        print(f"In {j}th interval [{x_range[0]}, {x_range[1]}]:\n")
        for i in range(len(peaks_positions)):  # i: the i-th set of data
            if len(peaks_positions[i][j]) == 1:  # element is -1: no value
                print(f"{i}th data set: ")
                print(f"no peak here\n")
            elif len(peaks_positions[i][j]) == 2:
                print(f"{i}th data set: ")
                print(f"peak position:{peaks_positions[i][j]}")
        j += 1


def check_input_format(x_ranges):
    """ If the input is a list of shape (2,), turn it to a tuple. 
    e.g. [1,1] --> [[1,1]]
    """
    if np.array(x_ranges).shape == (2,):
        temp = []
        temp.append(x_ranges)
        return temp
    else:
        return x_ranges


# 改注释
# 3. look for peaks in several intervals (by using function find_peaks())
def plot_peaks_in_ranges(x, ys, x_ranges):
    """ Draw a figure for the data read from file and mark the peaks for an 
    interval. 
    
    The inputs:
        x: 1-D array.
        ys: 2-D array.
        x_ranges: 2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range
    """
    x_ranges = check_input_format(x_ranges)

    peaks_positions = []
    for i in range(len(ys)):  # the i-th data set for y-axis
        current_y = ys[i]
        plt.loglog(x, current_y, linewidth = 0.6)  # the lines
        peak_indices, _ = find_peaks_in_ranges(x, current_y, x_ranges)

        current_peaks = []  # peaks for a data set in several intervals
        for j in range(len(x_ranges)):  # the j-th interval on x-axis
            peak_index = peak_indices[j]
            peak_x = x[peak_index]
            peak_y = current_y[peak_index]
            plt.loglog(peak_x, peak_y, ".")

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


def gaussian(x, h, xc, w): 
    """ The function to fit the curve.
    Return the y values resulted from gaussian function. 

    h: the height of the curve's peak
    xc: the position of the center of the peak
    w: the width of the curve
    """
    return h * np.exp(-(x-xc)**2/(2*(w**2)))


# 改备注 a test 找到一个范围的gaussian curve and print
def fit_gaussian(x, y, x_range):
#     """ Return parameters of func gaussian() needed to fit gaussian model, and
#      the standard deviation.

#     x: 1-D array
#     y: 1-D array
#     """
    interval_indices = indices_in_interval(x, x_range)
    x_in_interval = x[interval_indices]
    y_in_interval = y[interval_indices]
    
    popt, pcov = curve_fit(gaussian, x_in_interval, y_in_interval, maxfev = 1000000)
    error = np.sqrt(np.diag(pcov))  # standard deviation
    x_gaussian = np.linspace(x_in_interval[0], max(x_in_interval), 100)
    y_gaussian = gaussian(x_gaussian, *popt)

    gaussian_result = []
    gaussian_result.append(x_gaussian)
    gaussian_result.append(y_gaussian)
    # plt.plot(x, y, linewidth = 0.6)  # the lines
    # plt.plot(x_in_interval, y_gaussian, linewidth = 0.6, linestyle = ':')
    # plt.xlabel('2-theta') 
    # plt.ylabel('Intensity') 
    # plt.show()

    return popt, error, gaussian_result


# def check_peak(x, current_y, x_range):

    


# 对于要求找peak的所有地方，找到peak以及对应的高斯曲线
def fit_gaussian_full(x, ys, x_ranges):

    x_ranges = check_input_format(x_ranges)

    peaks_positions = []
    gauss_popts = []
    gauss_errors = []
    for i in range(len(ys)):  # the i-th data set for y-axis
        current_y = ys[i]
        plt.plot(x, current_y, linewidth = 0.6)  # the lines
        # peak_indices, peak_properties = find_peaks_in_ranges(x, current_y, 
        #                                 x_ranges)

        current_peaks = []  # peaks for a line
        current_gauss_popts = []  # parameters of gaussian for a line
        current_gauss_errors = []
        for x_range in x_ranges:  
            peak_index, _ = find_interval_peak(x, current_y, x_range)
            peak_x = x[peak_index]
            peak_y = current_y[peak_index]
            plt.plot(peak_x, peak_y, ".")

            current_peak = []  # the peak for an interval

            if len(peak_index) == 0: # no peak
                current_peak.append(-1)
                current_gauss_popts.append([-1, -1, -1])
                current_gauss_errors.append([-1, -1, -1])
            else:
                current_peak.append(peak_x[0])
                current_peak.append(peak_y[0])

                popt, error, gaussian_result = fit_gaussian(x, current_y, x_range)
                current_gauss_popts.append(popt)
                current_gauss_errors.append(error)
                plt.plot(gaussian_result[0], gaussian_result[1], linewidth = 0.8, linestyle = '--')
            current_peaks.append(current_peak)

        peaks_positions.append(current_peaks)
        gauss_popts.append(current_gauss_popts)
        gauss_errors.append(current_gauss_errors)

    plt.xlabel('2-theta') 
    plt.ylabel('Intensity') 
    plt.show()

    print_curve_fit(peaks_positions, gauss_popts, gauss_errors, x_ranges)


# 改注释
def print_curve_fit(peaks_positions, popts, errors, x_ranges):
    """ Print positions for peaks in an interval. 
    
    Inputs: 
        peaks_positions: 3-D array of 2 columns
        2-D list or array of 2 columns, 1st col: the minimum of range
                                                2nd col: maximum of range
    """
    j = 0  # j: the j-th interval
    for x_range in x_ranges:
        print(f"\n\nIn {j}th interval [{x_range[0]}, {x_range[1]}]:")
        for i in range(len(peaks_positions)):  # i: the i-th set of data
            if len(peaks_positions[i][j]) == 1:  # element is -1: no value
                print(f"\n{i}th data set: no peak here")
            elif len(peaks_positions[i][j]) == 2:
                print(f"\n{i}th data set: ")
                print(f"peak position:{peaks_positions[i][j]}")
                print(f"height = {'%.6f'%popts[i][j][0]} (+/-) {'%.6f'%errors[i][j][0]}")
                print(f"center = {'%.6f'%popts[i][j][1]} (+/-) {'%.6f'%errors[i][j][1]}")
                print(f"wideth = {'%.6f'%popts[i][j][2]} (+/-) {'%.6f'%errors[i][j][2]}")

        j += 1


def lorentz(x, y0, xc, w, a): 
    """ The function to fit the curve.
    Return the y values resulted from Lorentz function. 

    y0: offset
    xc: the position of the center of the peak
    w: the width of the curve
    a: area of the curve
    """
    pi = math.pi
    return y0 + (2*a/pi)*(w/(4*(x-xc)**2+w**2))

