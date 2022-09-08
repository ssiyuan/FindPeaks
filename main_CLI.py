import numpy as np

from import_files import (
    read_csv_dat,
    read_ascii_dir,
    process_original_data)

from fitting_peaks import (
    plot_initial_2d,
    plot_initial_3d,
    choose_model_with_str,
    summarize_data3D,
    summarize_peaks,
    summarize_comparison)

from validation import (
    is_valid_int_pos,
    is_valid_float_pos,
    is_valid_string_range,
    is_valid_x_unit)


def check_input_int(str_int):
    """Ensure the input can be converted to a positive integer.

    Args:
        str_int (string): the string input by user

    Returns:
        int: the integer format of the string
    """
    while not is_valid_int_pos(str_int):
        str_int = input("   Please input a valid integer number: ")
    return int(str_int)


def check_input_float(str_float):
    """Ensure the input can be converted to a positive float.

    Args:
        str_float (string): the string input by user

    Returns:
        float: the float format of the string
    """
    while not is_valid_float_pos(str_float):
        str_float = input("   Please input a valid float number: ")
    return float(str_float)


def check_answer_y_n(answer):
    """Check if the input is 'y' or 'n'.

    Args:
        answer (string): the string input by a user

    Returns:
        string: 'y' or 'n'
    """
    while answer != 'y' and answer != 'n':
        answer = input(
            "\n   Please choose a valid answer between 'y' and 'n': ")
    return answer


def input_file():
    """Get data in file(s) input by terminal.

    Returns:
        array: 2D array, first row as x, other rows as y datasets
    """
    data = []
    file_type = input("\n1. Input the type of file (1. csv/dat or 2. ASCII): ")

    if file_type == '1. csv/dat' or file_type == 'csv/dat' or file_type == '1'\
        or file_type == 'csv' or file_type == 'dat' or file_type == 'CSV' or \
            file_type == 'DAT':
        path = input("\n2. Input the file path: ")
        data = read_csv_dat(path)
    elif file_type == 'ASCII' or file_type == 'ascii' or file_type == '2':
        print("\n2. All ASCII files should be stored in a seperate directory.")
        path = input("   Input the path of directory storing ASCII files: ")
        data = read_ascii_dir(path)
    else:
        raise ValueError(f"Not a valid file type: {file_type}")

    return data


def plot_2d_3d_figures(x, ys):
    """Plot 2D and 3D figures of original data.

    Args:
        x (array): 1D
        ys (array): 2D, y datasets
    """
    print("\n3. Plot the 2d figure: ")
    plot_initial_2d(x, ys)
    figure_choice = input("   Plot the 3d figure? (y/n) ")
    if check_answer_y_n(figure_choice) == 'y':
        plot_initial_3d(x, ys)
    print_store_figures()


def input_peak_range(x):
    """Read range input from terminal.

    Returns:
        array: shape of (1,2), lower and upper bound for x
    """
    print("\n4. Start fitting: ")
    x_min, x_max = input(
        "   Input the range for peaks (x), seperate the 2 numbers with ',': ").split(',')
    while not is_valid_string_range(x, x_min, x_max):
        x_min, x_max = input(
            "   Please input valid range, e.g. 1.0,2.0: ").split(',')
    return np.array([float(x_min), float(x_max)])


def input_peak_num():
    """Read the number of peaks from terminal.

    Returns:
        int: number of peaks
    """
    peak_num = input("   Input number of peaks: ")
    return check_input_int(peak_num)


def input_pear_par_guess(peak_num):
    """Read initial guess for peaks from terminal.

    Args:
        peak_num (int): number of peaks

    Returns:
        array: shape of (peak_num, 3), each row for guess of parameters
    """
    peak_par_guess = np.zeros((peak_num, 3))
    print("\n5. Guess model parameters for peaks based on the 2d/3d figure. ")
    print("   (The 2d figure was stored in folder 'output_figures'.)")
    for i in range(peak_num):
        print(f"\n   Peak {i+1} pars: ")
        c = input("   (1) center (x): ")
        peak_par_guess[i][0] = check_input_float(c)
        s = input("   (2) sigma (characteristic width): ")
        peak_par_guess[i][1] = check_input_float(s)
        a = input("   (3) amplitude (overall intensity or area of peak): ")
        peak_par_guess[i][2] = check_input_float(a)
    return peak_par_guess


def input_center_min():
    """Read the lower bound from terminal.

    Returns:
        float: lower bound for peak center
    """
    center_min = 0
    min_choice = input("\n   Set a lower bound for the center? (y/n) ")
    if check_answer_y_n(min_choice) == 'y':
        center_min = input("   Input the lower bound for center: ")
        center_min = check_input_float(center_min)
    return float(center_min)


def input_index():
    """Read an index from terminal. The chosen dataset need to have all peaks. 

    Returns:
        int: an index of dataset
    """
    index = input(
        "\n6. Choose a dataset to compare Gaussian, Lorentzian and Pseudo-Voigt Models. Input the index: ")
    return check_input_int(index)


def input_directory():
    """Check if you want to store CSV files in a new folder.

    Returns:
        string (the path to store file) OR 1 (if no input path)
    """
    dir_choice = input(
        "   The default directory to store resulted data is 'output_files', do you want to change it? (y/n) ")
    if check_answer_y_n(dir_choice) == 'y':
        dir_path = input(
            "   Input a new directory path (Do not input '/' at the end.): ")
        return dir_path
    else:
        return 'output_files'


def input_model():
    """Read the model name from terminal.

    Returns:
        function: corresponding function for the model name input
    """
    model_choice = input(
        f"\n7. The default model is Gaussian Model, dou you want to change it? (y/n) ")
    if check_answer_y_n(model_choice) == 'n':
        return choose_model_with_str('Gaussian')
    else:
        model = input("   Choose a model with the first letter g/l/p: ")
        while choose_model_with_str(model) == 1:
            model = input("   The input is invalid. Use the first letter: ")
        Model = choose_model_with_str(model)
        return Model


def plot_changes(data_3d, dir_path):
    """Summarize the fitting results including changes in features along time.

    Args:
        data_3d (array): 1st dimension: peak
                        2nd dimension: time,amplitude,error,center,error,sigma,error,fwhm,error,height,error
                        3rd dimension: dataset
    """
    summary_changes = input("\n8. Summarize changes along time? (y/n) ")
    if summary_changes == 'y':
        # plot fwhm, plot intensity, tabulate results
        summarize_peaks(data_3d, dir_path)
        print_store_figures()


def print_store_data(dir_path):
    print(f"   The resulted data have been stored in folder '{dir_path}'. ")


def print_store_figures():
    print("   The resulted figures have been stored in folder 'output_figures'. ")


def input_x_unit():
    unit = input("\n   Select the unit for x (1. 2theta OR 2. q): ")
    while not is_valid_x_unit(unit):
        unit = input("   Please input a useful answer (1 or 2): ")


def main():
    data = input_file()
    # Number input: test_data/NH4OH-FAU-Practice-data.csv
    #  ASCII input: test_data/ASCII_data
    unit = input_x_unit()
    x, ys = process_original_data(data, unit)
    plot_2d_3d_figures(x, ys)
    x_range = input_peak_range(x)
    # Number input: 1,   6.5
    #  ASCII input: 1.4, 6.4
    peak_num = input_peak_num()
    # Number input: 2
    #  ASCII input: 5
    peak_par_guess = input_pear_par_guess(peak_num)
    # Number input: 6.35 0.038 0.00934
    #               1.8  0.2   0.003
    #  ASCII input: 5.9  0.01  1
    #               5.3  0.01  1    
    #               4.5  0.01  1
    #               3.9  0.01  1
    #               2.3  0.01  1
    center_min = input_center_min()
    index = input_index()
    # Number input: 11
    #  ASCII input: 8
    dir_path = input_directory()
    summarize_comparison(x, ys[index], x_range,
                         peak_par_guess, center_min, dir_path)
    print_store_data(dir_path)
    Model = input_model()
    data_3d = summarize_data3D(
        Model, x, ys, x_range, peak_par_guess, center_min)
    print_store_data(dir_path)
    print_store_figures()
    plot_changes(data_3d, dir_path)
    print("\n")


if __name__ == "__main__":
    main()
