from pathlib import Path

from utils import (
    read_molecule_data,
    separate_data,
    plot_initial_figure,
    plot_peaks,
    plot_maxes_in_range,
    plot_peaks_in_range,
    plot_peaks_in_ranges,
    fit_gaussian,
    fit_gaussian_full,
    fit_lorentz_full,
    fit_lorentz,
    try_sets)


def main():
    file_path = Path("Test_Data/NH4OH-FAU-Practice-data.csv")
    data = read_molecule_data(file_path)
    x, ys = separate_data(data)
    # plot_initial_figure(x, ys)
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
    try_sets(x, ys, [1.0, 3.0])
    

if __name__ == "__main__":
    main()
