import pandas as pd
import groups
import vectorize
from os import listdir
import re


all_files_path = "../Data/Verbal Tasks redux split/"
all_files = listdir(all_files_path)


def get_number(filename: str) -> str:
    pattern = r"(^\d+)"
    regex = re.compile(pattern)
    list_with_number = regex.findall(filename)
    return list_with_number[0]


def filter_files_by_groups(files: list, group1: str, group2: str) -> list:
    filtered_files = []
    for file in files:
        patient_number = get_number(file)
        patient_group = groups.number_to_group(patient_number)
        if patient_group == group1 or patient_group == group2:
            filtered_files.append(file)
    filtered_files.sort()
    return filtered_files


def filter_files_by_vftype(files: list, vf_type: str) -> list:
    return [file for file in files if vf_type in file]


def file_name_to_group(file_name: str):
    patient_number = get_number(file_name)
    patient_group = groups.number_to_group(patient_number)
    return patient_group


def vectors_dataframe(path: str, files: list):
    vectors = [vectorize.vectorize_vf(path + file) for file in files]
    df = pd.DataFrame(vectors)
    return df


def groups_series(file_list: list):
    groups_list = [file_name_to_group(file_name) for file_name in file_list]
    s = pd.Series(groups_list)
    return s


def relabel_groups(group_var: str, group1: str) -> int:
    return int(group_var == group1)


def filtered_features_and_target(group1: str, group2: str, vf_type: str):
    files = all_files
    filtered_files = filter_files_by_vftype(filter_files_by_groups(files, group1, group2), vf_type)
    X = vectors_dataframe(all_files_path, filtered_files)
    y = groups_series(filtered_files)
    # relabeling the groups to 1 and 0
    y = y.apply(lambda g: int(group1 == g))
    return X, y


def merge_function(x, merge_list):
    if x in merge_list:
        return 0
    else:
        return 1


def merge_groups(series, merge_list):
    merged_series = series.apply(lambda x: merge_function(x, merge_list))
    return merged_series


def filtered_features_and_target_merged_groups(merged_groups: list, vf_type: str):
    filtered_files = filter_files_by_vftype(all_files, vf_type)
    X = vectors_dataframe(all_files_path, filtered_files)
    y = groups_series(filtered_files)
    y = merge_groups(y, merged_groups)
    return X, y