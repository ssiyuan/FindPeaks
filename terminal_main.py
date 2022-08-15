import numpy as np

from OpenFile import (
    read_csv,
    read_ascii_files)

from utils import (
    process_original_data,
    plot_initial_2d,
    plot_initial_3d,
    choose_model_with_str,
    summarize_data3D,
    summarize_peaks,
    summarize_comparison)


def input_file():
    data = []
    file_type = input('\n1. Input the type of file (1. csv/dat or 2. ASCII): ')

    if file_type == '1. csv/dat' or file_type == 'csv/dat' or file_type == '1'\
        or file_type == 'csv' or file_type == 'dat':
        path = input('\n2. Input the file path: ')
        data = read_csv(path)

    if file_type == 'ASCII' or file_type == 'ascii' or file_type == '2':
        print('\n2. All ASCII files should be stored in a seperate directory.')
        path = input('   Input the path of directory storing ASCII files: ')
        data = read_ascii_files(path)

    return data


def plot_2d_3d_figures(x, ys):
    print('\n3. Plot the 2d figure: ')
    plot_initial_2d(x, ys)

    figure_choice = input('\n   Plot the 3d figure? (y/n) ')
    while figure_choice != 'y' and figure_choice != 'n':
        figure_choice = input('\n   Please choose a valid answer: ')
    if figure_choice == 'y':
        plot_initial_3d(x, ys)


def input_peak_range():
    print('\n4. Start fitting: ')
    x_min, x_max = input('\n   Input the range for peaks: ').split(',')
    x_range = np.array([float(x_min), float(x_max)])
    return x_range


def input_peak_num():
    peak_num = int(input('\n   Input number of peaks: '))
    return peak_num


def input_pear_par_guess(peak_num):
    peak_par_guess = np.zeros((peak_num,3))
    print('\n5. Input guess for parameters of peaks\n\n   5.1')
    for i in range(peak_num):
        print('\n   Peak {} pars: '.format(i+1))
        c = input('   center (x): ')
        s = input('   sigma (characteristic width): ')
        a = input('   amplitude (overall intensity or area of peak): ')
        peak_par_guess[i][0] = float(c)
        peak_par_guess[i][1] = float(s)
        peak_par_guess[i][2] = float(a)
    return peak_par_guess


def input_center_min():
    center_min = 0
    min_choice = input('\n   5.2 Set a lower bound for the center? (y/n) ')
    while min_choice != 'y' and min_choice != 'n':
        min_choice = input('\n   Please choose a valid answer: ')
    if min_choice == 'y':
        center_min = input('\n   Input the lower bound for center: ')
        center_min = float(center_min)
    return center_min


def input_index():
    index = input('\n6. Choose a dataset to compare Gaussian, Lorentzian and \
Pseudo-Voigt Models: ')
    return int(index)


def input_model():
    model = input('\n7. Choose a model with the first letter g/l/p: ')
    while choose_model_with_str(model) == 1:
        model = input('\n   The input is invalid. Use the first letter: ')
    Model = choose_model_with_str(model)
    return Model


def plot_changes(data_3d):
    summary_changes = input('\n8. Summarize changes along time? (y or n) ')
    if summary_changes == 'y':
        summarize_peaks(data_3d)  # plot_fwhm, plot_intensity, tabulate_result




def main():
    data = input_file()
    # Test_Data/NH4OH-FAU-Practice-data.csv
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
    summarize_comparison(x, ys[index], x_range, peak_par_guess, center_min)

    Model = input_model()
    data_3d = summarize_data3D(Model, x, ys, x_range, peak_num, peak_par_guess\
        , center_min)
    plot_changes(data_3d)


if __name__ == "__main__":
    main()
