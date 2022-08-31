"""Functions to read files including single CSV/DAT file, list of ASCII files, and a 
folder only including ASCII files.
"""

import csv
import codecs
import os

import numpy as np

from pathlib import Path
from validation import (
    validate_file_data,
    is_valid_file,
    is_valid_dir)


def check_file_type(file_path):
    """Check if the file is CSV or DAT format. """
    is_valid_file(file_path)
    if file_path[-4:] != '.csv' and file_path[-4:] != '.dat':
        raise TypeError(
            f"The format of input file should be CSV or DAT: {file_path}")
    return file_path[-4:]


def read_csv_dat(file_path):
    """Read a CSV or DAT file and transpose the data in the file. """
    file_type = check_file_type(file_path)
    file_path = Path(file_path)
    data, file_reader = [], []
    with file_path.open(mode="r", encoding="utf-8") as file:
        if file_type == '.csv':
            file_reader = csv.reader(file, delimiter=",", quotechar='\"')
        elif file_type == '.dat':
            file_reader = csv.reader(file, delimiter="\t", quotechar='\"')
        for line in file_reader:
            data.append(line)

    # The data in file is read by column, so transpose it to read by row.
    data_needed = np.array(data).transpose()
    data_needed = data_needed.astype(np.float32)
    return data_needed


def read_ascii(file_path):
    """Read a file where data is stored in ASCII format and transpose the data. """
    is_valid_file(file_path)
    with codecs.open(file_path, mode="r", encoding="utf-8-sig") as file:
        data_set = np.loadtxt(file, skiprows=26, dtype=float)
        return data_set.transpose()


def check_dir_end(dir_path):
    """Check if the input directory path ends with '/'. If it does, delete extra '/'. """
    if dir_path[-1] != '/':
        return dir_path
    else:
        dir_path = dir_path[:-1]
        return check_dir_end(dir_path)


def read_ascii_files(file_paths):
    """Read ASCII files. 

    Args:
        file_paths (list): a list of strings corresponding to file paths

    Returns:
        array: 2D array, first row as x dataset and other rows as y datasets
    """
    file_paths.sort()
    data = []
    i = 0
    # Use i to check the index of current file, store the first column as x
    # only when i = 0
    for file_path in file_paths:
        data_read = read_ascii(file_path)
        if i == 0:
            data.append(data_read[0])  # x
        data.append(data_read[1])  # y-s
        i += 1
    return np.array(data).astype(np.float32)


def read_ascii_dir(dir_path):
    """Read ASCII files in a seperate directory. 

    Args:
        dir_path (string): the path to the folder only storing ASCII files

    Returns:
        array: 2D array, first row as x values and other rows storing y-s
    """
    dir_path = check_dir_end(dir_path)
    is_valid_dir(dir_path)
    file_names = os.listdir(dir_path)
    file_names.sort()

    data = []
    i = 0
    # Use i to check the index of current file, store the first column as x
    # only when i = 0
    for file_name in file_names:
        if file_name != ".DS_Store":
            file_path = dir_path + "/" + file_name
            data_read = read_ascii(file_path)
            if i == 0:
                data.append(data_read[0])  # x
            data.append(data_read[1])  # y-s
            i += 1
    return np.array(data).astype(np.float32)


def check_output_dir(dir_path):
    """Check if the input directory path is valid. Create one if invalid.

    Args:
        dir_path (string): path of directory to store file
    """
    check_dir_end(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def process_original_data(file_data):
    """Seperate the x values and datasets of y stored in input array.

    Args:
        file_data (array): 2D array, the first column storing x-axis, other columns storing y-axis.

    Returns:
        array: 1D
        array: 2D, each row as a seperate dataset of y
    """
    validate_file_data(file_data)
    x = file_data[0]  # 2-theta
    ys = file_data[1:]  # intensity
    return x, ys