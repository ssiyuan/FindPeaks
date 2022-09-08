"""Test the time needed for running the programme. """

from lmfit.models import GaussianModel

from import_files import (
    read_csv_dat,
    process_original_data)

from fitting_peaks import (
    plot_initial_3d,
    summarize_data3D,
    summarize_peaks,
    form_mesopore)


def get_mesopore_extent(file_name):
    data = read_csv_dat(file_name)
    x, ys = process_original_data(data,'2theta')
    plot_initial_3d(x, ys, index_min=40, index_max=600)
    csv_guess1 = [4.5, 0.029, 0.007]
    csv_guess2 = [1.2, 0.14, 0.002]
    data_3d = summarize_data3D(GaussianModel, x,ys,[0.9,4.65],[csv_guess1,csv_guess2], center_min=1.0)
    summarize_peaks(data_3d)
    form_mesopore(data_3d)


def main():
    # numerical data
    get_mesopore_extent("Zeolite/cbv712.dat")
    get_mesopore_extent("Zeolite/cbv720.dat")
    get_mesopore_extent("Zeolite/cbv760.dat")
    get_mesopore_extent("Zeolite/cbv901.csv")


if __name__ == "__main__":
    main()