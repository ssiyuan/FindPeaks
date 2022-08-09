import numpy as np

from pathlib import Path

from utils import (
    read_csv,
    read_ascii_files,
    process_original_data,
    plot_initial_2d,
    plot_initial_3d,
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

    figure_choice = input('\n3. Choose a figure of initial data (2d or 3d): ')
    if figure_choice == '2d' or figure_choice == '2':
        plot_initial_2d(x, ys)
    if figure_choice == '3d' or figure_choice == '3':
        plot_initial_3d(x, ys)
    
    
    x_min, x_max = input('\nInput the range for peaks: ').split(',')
    x_range = np.array([float(x_min), float(x_max)])
    # [1,6.5]
    # [2,5]
    peak_num = int(input('\nInput number of peaks: '))
    # 2
    # 3
    peak_par_guess = np.zeros((peak_num,3))
    for i in range(peak_num):
        c,s,a=input('\nInput guess for Peak {} pars: '.format(i+1)).split(',')
        peak_par_guess[i][0] = float(c)
        peak_par_guess[i][1] = float(s)
        peak_par_guess[i][2] = float(a)
    # print(peak_par_guess)
    # print(type(peak_par_guess))
    # # [[6.35, 0.038, 0.00934], [1.8, 0.2, 0.003]]
    # [[4.5, 0.07, 1.2], [3.82, 0.07, 1.13], [2.4, 0.038, 0.3]]
    data_3d = summarize_data3D(x, ys, x_range, peak_num, peak_par_guess)

    summary_changes = input('\n4. Summarize changes along time? (y or n) ')
    if summary_changes == 'y':
        summarize_peaks(data_3d)  # plot_fwhm, plot_intensity, tabulate_result

    summary_comparison = input('\n5. Compare between 3 function? (y or n) ')
    dataset_index = input('\n   Please choose a dataset for the comparison: ')
    # 11
    # 8
    if summary_comparison == 'y':
        summarize_comparison(x, ys[int(dataset_index)], x_range, peak_par_guess)


    # # test_data
    # data = read_csv("Test_Data/NH4OH-FAU-Practice-data.csv")
    # x, ys = process_original_data(data)
    # plot_initial_3d(x, ys)
    # plot_initial_2d(x, ys)
    # # csv_guess1 = [6.35, 0.038, 0.00934]
    # # csv_guess2 = [1.8, 0.2, 0.003]
    # # data_3d = summarize_data3D(x,ys,[1,6.5],2,csv_guess1,csv_guess2)    
    # # summarize_peaks(data_3d)
    # # summarize_comparison(x, ys[11], [1,6.5], csv_guess1, csv_guess2)
    # csv_guess = [[6.35, 0.038, 0.00934],[1.8, 0.2, 0.003]]
    # summarize_comparison(x, ys[11], [1,6.5], csv_guess)


    # # ASCII_data
    # data = read_ascii_files("ASCII_data")
    # x, ys = process_original_data(data)
    # plot_initial_3d(x, ys)
    # plot_initial_2d(x, ys)
    # # ascii_guess2 = [2.4, 0.038, 0.3]
    # # ascii_guess1 = [3.82, 0.07, 1.13]
    # ascii_guess = [[4.5, 0.07, 1.2], [3.82, 0.07, 1.13], [2.4, 0.038, 0.3]]
    # # data_3d = summarize_data3D(x,ys,[2, 4],2,ascii_guess1,ascii_guess2)
    # # summarize_peaks(data_3d)
    # summarize_comparison(x, ys[8], [2, 5], ascii_guess)
    

if __name__ == "__main__":
    main()
