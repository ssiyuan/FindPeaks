import numpy as np

from OpenFile import (
    read_csv,
    read_ascii_files)

from utils import (
    process_original_data,
    plot_initial_2d,
    plot_initial_3d,
    choose_model,
    summarize_data3D,
    summarize_peaks,
    summarize_comparison)


def main():
    data = []
    file_type = input('\n1. Input the type of file (csv or ASCII): ')
    if file_type == 'csv' or file_type == '.csv' or file_type == 'CSV':
        path = input('\n2. Input the .csv file path: ')
        data = read_csv(path)

    if file_type == 'ASCII' or file_type == 'ascii':
        print('\n2. All ASCII files should be stored in a seperate directory.')
        path = input('   Input the path of directory storing ASCII files: ')
        data = read_ascii_files(path)

    # Test_Data/NH4OH-FAU-Practice-data.csv
    # ASCII_data

    x, ys = process_original_data(data)

    print('\n3. Plot the 2d figure: ')
    plot_initial_2d(x, ys)
    figure_choice = input('\n   Plot the 3d figure? (y/n) ')
    while figure_choice != 'y' and figure_choice != 'n':
        figure_choice = input('\n   Please choose a valid answer: ')
    if figure_choice == 'y':
        plot_initial_3d(x, ys)
    
    
    print('\n4. Start fitting: ')
    x_min, x_max = input('\n   Input the range for peaks: ').split(',')
    x_range = np.array([float(x_min), float(x_max)])
    # Input: 1,6.5
    # Input: 2,5
    peak_num = int(input('\n   Input number of peaks: '))
    # Input: 2
    # Input: 3
    

    peak_par_guess = np.zeros((peak_num,3))
    print('\n5. Input guess for parameters of peaks')
    for i in range(peak_num):
        print('\n   Peak {} pars: '.format(i+1))
        c = input('   center (x): ')
        s = input('   sigma (characteristic width): ')
        a = input('   amplitude (overall intensity or area of peak): ')
        peak_par_guess[i][0] = float(c)
        peak_par_guess[i][1] = float(s)
        peak_par_guess[i][2] = float(a)
    # # Input: 6.35 0.038 0.00934
    # 1.8 0.2 0.003
    # Input: 4.5 0.07 1.2
    # 3.82 0.07 1.13
    # 2.4 0.038 0.3

    index = input('\n5. Choose a dataset to compare Gaussian, Lorentzian and \
Pseudo-Voigt Models: ')
    # 11
    # 8
    summarize_comparison(x, ys[int(index)], x_range, peak_par_guess)
    model = input('\n   Choose a model with the first letter g/l/p: ')
    while choose_model(model) == 1:
        model = input('\n   The input is invalid. Please use the first letter: ')
    Model = choose_model(model)

    data_3d = summarize_data3D(Model, x, ys, x_range, peak_num, peak_par_guess)

    summary_changes = input('\n6. Summarize changes along time? (y or n) ')
    if summary_changes == 'y':
        summarize_peaks(data_3d)  # plot_fwhm, plot_intensity, tabulate_result


if __name__ == "__main__":
    main()
