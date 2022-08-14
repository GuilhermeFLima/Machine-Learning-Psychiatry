# Takes verbal tasks csv files and splits entries with
# multiple words into different entries, with one word each.


import pandas as pd
from os import listdir


def entry_split(series):
    split_list = []
    for e in series:
        split_list += e.split('_')

    return pd.Series(split_list)


def split_all_in_folder(open_path, save_path):
    all_files = listdir(open_path)
    csv_files = [x for x in all_files if '.csv' in x]
    for file in csv_files:
        df = pd.read_csv(open_path + file)
        s = df['0']
        s = entry_split(s)
        s.to_csv(save_path + file, index=False)

    return None


if __name__ == '__main__':
    path_to_verbal_tasks_folder = "../Data/Verbal Tasks redux/"
    path_to_split_folder = "../Data/Verbal Tasks redux split/"
    split_all_in_folder(open_path=path_to_verbal_tasks_folder, save_path=path_to_split_folder)



