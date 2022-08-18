import numpy as np

from import_files import (
    read_csv_dat,
    read_ascii_dir)

from analysis import (
    process_original_data,
    plot_initial_2d,
    plot_initial_3d,
    choose_model_with_str,
    summarize_data3D,
    summarize_peaks,
    summarize_comparison)


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

    figure_choice = input("\n   Plot the 3d figure? (y/n) ")
    while figure_choice != 'y' and figure_choice != 'n':
        figure_choice = input("\n   Please choose a valid answer: ")
    if figure_choice == 'y':
        plot_initial_3d(x, ys)

    print_store_figures()


def input_peak_range():
    """Read range input from terminal.

    Returns:
        array: shape of (1,2), lower and upper bound for x
    """
    print("\n4. Start fitting: ")
    x_min, x_max = input("\n   Input the range for peaks (x): ").split(',')
    x_range = np.array([float(x_min), float(x_max)])
    return x_range


def input_peak_num():
    """Read the number of peaks from terminal.

    Returns:
        int: number of peaks
    """
    peak_num = int(input("\n   Input number of peaks: "))
    return peak_num


def input_pear_par_guess(peak_num):
    """Read initial guess for peaks from terminal.

    Args:
        peak_num (int): number of peaks

    Returns:
        array: shape of (peak_num, 3), each row for guess of parameters
    """
    peak_par_guess = np.zeros((peak_num, 3))
    print("\n5. Input guess for parameters of peaks\n\n   5.1")
    for i in range(peak_num):
        print(f"\n   Peak {i+1} pars: ")
        c = input("   center (x): ")
        s = input("   sigma (characteristic width): ")
        a = input("   amplitude (overall intensity or area of peak): ")
        peak_par_guess[i][0] = float(c)
        peak_par_guess[i][1] = float(s)
        peak_par_guess[i][2] = float(a)
    return peak_par_guess


def input_center_min():
    """Read the lower bound from terminal.

    Returns:
        float: lower bound for peak center
    """
    center_min = 0
    min_choice = input("\n   5.2 Set a lower bound for the center? (y/n) ")
    while min_choice != 'y' and min_choice != 'n':
        min_choice = input("\n   Please choose a valid answer: ")
    if min_choice == 'y':
        center_min = float(input("\n   Input the lower bound for center: "))
    return center_min


def input_index():
    """Read an index from terminal.

    Returns:
        int: an index of dataset
    """
    index = input(
        "\n6. Choose a dataset to compare Gaussian, Lorentzian and Pseudo-Voigt Models: ")
    return int(index)


def input_directory():
    """Check if you want to store CSV files in a new folder.

    Returns:
        string (the path to store file) OR 1 (if no input path)
    """
    dir_choice = input(
        "\n   The default directory to store resulted data is 'output_files', do you want to change it? (y/n) ")
    while dir_choice != 'y' and dir_choice != 'n':
        dir_choice = input("\n   Please choose a valid answer: ")
    if dir_choice == 'y':
        dir_path = input(
            "\n   Input a new directory path (Do not input '/' at the end.): ")
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
    while model_choice != 'y' and model_choice != 'n':
        model_choice = input("\n   Please choose a valid answer: ")
    if model_choice == 'n':
        return choose_model_with_str('Gaussian')
    else:
        model = input("\n   Choose a model with the first letter g/l/p: ")
        while choose_model_with_str(model) == 1:
            model = input("\n   The input is invalid. Use the first letter: ")
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
        # plot_fwhm, plot_intensity, tabulate_result
        summarize_peaks(data_3d, dir_path)
        print_store_figures()


def print_store_data(dir_path):
    print(f"\nThe resulted data have been stored in folder '{dir_path}'. ")


def print_store_figures():
    print("\nThe resulted figures have been stored in folder 'output_figures'. ")


def main():
    data = input_file()
    # test_data/NH4OH-FAU-Practice-data.csv
    # ASCII_data
    x, ys = process_original_data(data)
    plot_2d_3d_figures(x, ys)
    x_range = input_peak_range()
    # Input 1: 1,6.5
    # Input 2: 2,5
    peak_num = input_peak_num()
    # Input 1: 2
    # Input 2: 3
    peak_par_guess = input_pear_par_guess(peak_num)
    # Input 1: 6.35 0.038 0.00934
    # 1.8 0.2 0.003
    # Input 2: 4.5 0.07 1.2
    # 3.82 0.07 1.13
    # 2.4 0.038 0.3
    center_min = input_center_min()
    index = input_index()  # the index of a data set with all peaks needed
    # Input 1: 11
    # Input 2: 8
    dir_path = input_directory()
    summarize_comparison(x, ys[index], x_range,
                         peak_par_guess, center_min, dir_path)
    print_store_data(dir_path)
    Model = input_model()
    data_3d = summarize_data3D(
        Model, x, ys, x_range, peak_num, peak_par_guess, center_min)
    print_store_data(dir_path)
    print_store_figures()
    plot_changes(data_3d, dir_path)


if __name__ == "__main__":
    main()
