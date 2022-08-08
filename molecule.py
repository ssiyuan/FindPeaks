from pathlib import Path

from utils import (
    read_csv,
    read_ascii_files,
    process_original_data,
    plot_initial_2d,
    plot_initial_3d,
    plot_peaks,
    plot_maxes_in_range,
    plot_peaks_in_range,
    plot_peaks_in_ranges,
    fit_gaussian,
    fit_gaussian_full,
    fit_lorentz_full,
    fit_lorentz,
    summarize_data3D,
    summarize_peaks,
    summarize_comparison)


def main():
    # plot_peaks(x, ys)
    # plot_peaks_in_range(x, ys, [0.725,0.875])  # data after log
    # plot_peaks_in_range(x, ys, [1, 2.8])
    # plot_maxes_in_range(x, ys, [1, 2.8])
    # plot_peaks_in_ranges(x, ys, [[1, 2.8], [6, 7]])
    # plot_peaks_in_ranges(x, ys, [0.2, 0.29])
    # fit_gaussian(x, ys[15], [1.2, 2.5])  # original data
    # fit_gaussian_full(x, ys, [[1.2, 2.5]]) 
    # fit_lorentz_full(x, ys, [[1.2, 2.5], [6.2, 6.5]])
    # try_sets(x, ys, [6.2, 6.5])
    # try_sets(x, ys, [1.0, 3.0])
    # plot_baseline(x, ys[14], [1.2, 2.5])

    # data = read_csv("Test_Data/NH4OH-FAU-Practice-data.csv")
    # x, ys = process_original_data(data)
    # plot_initial_3d(x, ys)
    # plot_initial_2d(x, ys)
    # csv_guess1 = [6.35, 0.038, 0.00934]
    # csv_guess2 = [1.8, 0.2, 0.003]
    # data_3d = summarize_data3D(x,ys,[1,6.5],2,csv_guess1,csv_guess2)    
    # summarize_peaks(data_3d)
    # summarize_comparison(x, ys[11], [1,6.5], csv_guess1, csv_guess2)

    data = read_ascii_files("ASCII_data")
    x, ys = process_original_data(data)
    plot_initial_3d(x, ys)
    plot_initial_2d(x, ys)
    ascii_guess1 = [2.4, 0.038, 0.3]
    ascii_guess2 = [3.82, 0.07, 1.13]
    # data_3d = summarize_data3D(x,ys,[2, 4],2,ascii_guess1,ascii_guess2)
    # summarize_peaks(data_3d)
    summarize_comparison(x, ys[8], [2, 4], ascii_guess1, ascii_guess2)


    # print(data)
    

if __name__ == "__main__":
    main()
