from pathlib import Path

from utils import (
    read_molecule_data,
    separate_data)


def main():
    file_path = Path("Test_Data/NH4OH-FAU-Practice-data.csv")
    data = read_molecule_data(file_path)
    x, ys = separate_data(data)
    

if __name__ == "__main__":
    main()
