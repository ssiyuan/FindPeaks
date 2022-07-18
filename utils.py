""" Utility functions for working with molecule data. """

import csv

import numpy as np


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
    print(x)
    print(ys)
    return x, ys


