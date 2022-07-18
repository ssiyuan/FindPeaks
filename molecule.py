from pathlib import Path

from utils import (
    read_molecule_data)


def main():
    file_path = Path("Test_Data/NH4OH-FAU-Practice-data.csv")
    data = read_molecule_data(file_path)


if __name__ == "__main__":
    main()
