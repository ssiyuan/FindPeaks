import csv
import codecs
import os

import numpy as np

from pathlib import Path


def check_file_type(file_path):
    # # validation
    # if file_path[-4:] != '.csv' and file_path[-4:] != '.dat':
    #     return TypeError
    return file_path[-4:]


def read_csv(file_path):
    """ Return the data from file_path, and convert to the right format.
        The input should be the path to a .csv file.
        only .csv/.dat
    """
    file_type = check_file_type(file_path)
    file_path = Path(file_path)
    data = []
    with file_path.open(mode='r', encoding="utf-8") as file:
        if file_type == '.csv':
            csv_reader = csv.reader(file, delimiter=",", quotechar='\"')
        elif file_type == '.dat':
            csv_reader = csv.reader(file, delimiter="\t", quotechar='\"')
        else:
            return "Error here."
        for line in csv_reader:
            data.append(line)

    # The data in file is read by column, so transpose it to read by row.
    data_needed = np.array(data).transpose()
    data_needed = data_needed.astype(np.float32)  # Convert string to float.
    return data_needed


def read_ascii(file_path):
    # f = open(file_path, 'r')
    # for line in f:
    #     print(repr(line))
    with codecs.open(file_path, mode='r', encoding="utf-8-sig") as file:
        data_set = np.loadtxt(file, skiprows=25, dtype=float)
        return data_set.transpose()


def check_dir_name(dir_path):
    # 文件路径末尾无"/"
    if dir_path[-1] != '/':
        return dir_path
    else:
        dir_path = dir_path[:-1]
        return check_dir_name(dir_path)


def read_ascii_files(dir_path):
    # check if it ends without '/'
    dir_path = check_dir_name(dir_path)
    data = []
    if os.path.isdir(dir_path):  # check if it is a directory
        files = os.listdir(dir_path)
        files.sort()
        i = 0  # check the index of current file, add x while reading the first
        for file_path in files:
            if file_path != ".DS_Store":
                file_path = dir_path + "/" + file_path
                data_read = read_ascii(file_path)
                if i == 0:
                    data.append(data_read[0])  # x
                data.append(data_read[1])  # y-s
                i += 1
    return np.array(data).astype(np.float32)