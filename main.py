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
    print("Input the type of file: ")
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
    data = read_ascii_files("ASCII_data")
    x, ys = process_original_data(data)
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
