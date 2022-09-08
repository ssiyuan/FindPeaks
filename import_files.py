"""Functions to read files including single CSV/DAT file, list of ASCII files, and a 
folder only including ASCII files.
"""

import csv
import codecs
import os
import math

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


def read_ascii(file_path, start_line = 26):
    """Read a file where data is stored in ASCII format and transpose the data. """
    is_valid_file(file_path)
    with codecs.open(file_path, mode="r", encoding="utf-8-sig") as file:
        data_set = np.loadtxt(file, skiprows=start_line, dtype=float)
        return data_set.transpose()


def check_dir_end(dir_path):
    """Check if the input directory path ends with '/'. If it does, delete extra '/'. """
    if dir_path[-1] != '/':
        return dir_path
    else:
        dir_path = dir_path[:-1]
        return check_dir_end(dir_path)


def read_ascii_files(file_paths, start_line=26):
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
        data_read = read_ascii(file_path, start_line)
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


def get_q(x, wavelength=0.15406):
    """Calculate q with 2-theta.

    Args:
        x (array): 1D array for 2-theta
        wavelength (float): Defaults to 0.15406, the wavelength of Cu-Kα source.

    Returns:
        array: q = 4 * pi * sin(theta) / wavelength
    """    
    radians = np.deg2rad(0.5*x)
    return 4 * math.pi * np.sin(radians) / wavelength


def log_ys(ys):
    """Get the log format for input 2D array.

    Args:
        ys (array): 2D array

    Returns:
        array: 2D array for log values of input
    """
    log_y = np.zeros((len(ys),len(ys[0])))
    for i in range(len(ys)):
        curr_y = ys[i]
        curr_y[curr_y==0] = 0.00001
        log_y[i] = np.log(curr_y)
    return log_y


def process_original_data(file_data):
    """Seperate the x values and datasets of y stored in input array.

    Args:
        file_data (array): 2D array, the first column storing x-axis, other columns storing y-axis.

    Returns:
        array: 1D, transfer 2-theta to q
        array: 2D, each row as a seperate dataset of y, transfer to log(y)
    """
    validate_file_data(file_data)
    x = get_q(file_data[0])  # 2-theta to q
    ys = log_ys(file_data[1:])  # log format of intensity
    return x, ys