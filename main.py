from pathlib import Path

from clear_utils import (
    read_data)


def main():
    file_path = Path("Test_Data/NH4OH-FAU-Practice-data.csv")
    data = read_data(file_path)
    x, ys = separate_data(data)


if __name__ == "__main__":
    main()