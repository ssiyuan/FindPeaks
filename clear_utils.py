""" Utility functions for working with molecule data. (make it more clear)""" 

import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def read_data(file_path):
    """ Return the data from file_path, and convert to the right format.
        The input should be the path to a .csv file.
    """
    data = []
    with file_path.open(mode='r', encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",", quotechar='\"')
        for line in csv_reader:
            data.append(line)

    # The data in file is read by column, so transpose it to read by row.
    transpose_data = np.array(data).transpose()
    transpose_data = transpose_data.astype(np.float32)  # Convert string to float.
    return transpose_data


def transpose_data(data):
    
