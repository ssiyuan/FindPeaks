""" Functions to validate inputs. """

import os.path

import numpy as np


def is_valid_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The input file does not exist: {file_path}")


def is_valid_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(
            f"The input directory does not exist: {dir_path}")


def is_valid_1D_array(array_1d):
    """Check if the input is 1D array. """
    return isinstance(array_1d, (np.ndarray, list)) and len(array_1d.shape) == 1


def is_valid_2D_array(array_2d):
    """Check if the input is 2D array. """
    return isinstance(array_2d, (np.ndarray, list)) and len(array_2d.shape) == 2


def is_valid_3D_array(array_3d):
    """Check if the input is 3D array. """
    return isinstance(array_3d, (np.ndarray, list)) and len(array_3d.shape) == 3


def validate_file_data(file_data):
    if not is_valid_2D_array(file_data):
        raise TypeError(
            f"Input array should be 2D array. {file_data} is {len(file_data.shape)}D {type(file_data)}. ")


def validate_x_range(x, x_range):
    """Written for calling the function from code. """
    if isinstance(x_range, list):
        x_range = np.array(x_range)
    if x_range.shape != (2,):
        raise ValueError(f"The input x_range should have shape of (2,): {x_range}. ")
    if x_range[0] > x_range[1]:
        raise ValueError(f"The input x_range should have smaller value as first element, and larger one as second: {x_range}. ")
    if x_range[0] < min(x) or x_range[1] > max(x):
        raise ValueError(f"The input range is inappropriate: {x_range}. ")


def is_valid_string_range(x, x_min, x_max):
    """Written for the Command Line Interface. """
    if not is_valid_float_pos(x_min):
        return False
    if not is_valid_float_pos(x_max):
        return False
    x_min = float(x_min)
    x_max = float(x_max)
    if x_min > x_max:
        return False
    if x_min < min(x) or x_max > max(x):
        return False
    return True


def is_valid_int(str_int):
    """Check if the input can be converted to a positive integer.

    Args:
        str_int (string): a string of integer

    Returns:
        bool: True if can be converted to integer, else False. 
    """
    try: 
        int(str_int)
        return True
    except ValueError:
        return False


def is_valid_int_pos(str_int):
    return is_valid_int(str_int) and int(str_int) > 0
        

def is_valid_float(str_float):
    """Check if the input can be converted to a positive float.

    Args:
        str_float (string): a string of float

    Returns:
        bool: True if can be converted to float, else False. 
    """
    try: 
        float(str_float)
        return True
    except ValueError:
        return False


def is_valid_float_pos(str_float):
    return is_valid_float(str_float) and float(str_float) > 0