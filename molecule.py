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
    fit_gaussian_full)


def main():
    file_path = Path("Test_Data/NH4OH-FAU-Practice-data.csv")
    data = read_molecule_data(file_path)
    x, ys = separate_data(data)
    # plot_initial_figure(x, ys)
    # plot_peaks(x, ys)
    # plot_peaks_in_range(x, ys, [0.725,0.875])  # data after log
    # plot_peaks_in_range(x, ys, [0.2, 0.29])
    # plot_maxes_in_range(x, ys, [0.2, 0.29])
    # plot_peaks_in_ranges(x, ys, [[0.2, 0.29], [0.725,0.875]])
    # plot_peaks_in_ranges(x, ys, [0.2, 0.29])
    # fit_gaussian(x, ys[0], [6.2, 6.5])  # original data
    fit_gaussian_full(x, ys, [[1.5, 2.0], [6.3, 6.43]]) 
    # fit_gaussian_full(x, ys, [[6.3, 6.43]]) 


if __name__ == "__main__":
    main()
