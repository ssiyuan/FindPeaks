""" Functions to validate inputs. """

import os.path


def is_valid_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The input file does not exist: {file_path}")


def is_valid_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The input directory does not exist: {dir_path}")



# def is_valid_code(code):
#     return isinstance(code, int) and 0 <= code <= 999


# def is_valid_name(name):
#     return isinstance(name, str) and 0 < len(name) <= 60


# def is_valid_id(glacier_id):
#     return (
#         isinstance(glacier_id, str)
#         and len(glacier_id) == 5
#         and all(digit in string.digits for digit in glacier_id)
#     )


# def is_valid_unit(unit):
#     return isinstance(unit, str) and len(unit) == 2 and (
#         all(char in string.ascii_letters.upper() for char in unit)
#         or unit == '99'
#     )


# def is_valid_location(lat, lon):
#     return -90 <= lat <= 90 and -180 <= lon <= 180


# def validate_info(glacier_id, name, unit, lat, lon, code):
#     """Check that the basic attributes of a glacier have reasonable values."""
#     if not is_valid_name(name):
#         raise ValueError(f"Inappropriate name for glacier: {name}")
#     if not is_valid_code(code):
#         raise ValueError(f"Inappropriate code for glacier: {code}")
#     if not is_valid_id(glacier_id):
#         raise ValueError(f"Inappropriate id for glacier: {glacier_id}")
#     if not is_valid_unit(unit):
#         raise ValueError(f"Inappropriate political unit for glacier: {unit}")
#     if not is_valid_location(lat, lon):
#         raise ValueError(f"Inappropriate location for glacier: {(lat, lon)}")


# def is_valid_code_pattern(code_pattern):
#     return (
#         is_valid_code(code_pattern)
#         or (
#             isinstance(code_pattern, str)
#             and len(code_pattern) == 3
#             and all(digit in string.digits + '?' for digit in code_pattern)
#         )
#     )


# def is_valid_balance(mass_balance):
#     return isinstance(mass_balance, float)


# def is_valid_year_type(year):
#     return isinstance(year, int)


# def is_valid_year_value(year):
#     return year <= datetime.date.today().year


# def validate_mass_balance_measurement(year, mass_balance):
#     if not is_valid_balance(mass_balance):
#         raise TypeError(f"Inappropriate mass balance type for glacier: {type(mass_balance)}")
#     if not is_valid_year_type(year):
#         raise TypeError(f"Inappropriate year type for glacier: {type(year)}")
#     if not is_valid_year_value(year):
#         raise ValueError(f"Inappropriate year value for glacier: {year}")

