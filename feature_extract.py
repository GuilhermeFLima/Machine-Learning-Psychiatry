import get_vec as gv
import neighbour_similarity as neighsim
import global_similarity as globalsim
import word_repetitions as wordrep
import groups
import pandas as pd
import numpy as np
import re
from os import listdir


def get_number(filename: str) -> str:
    pattern = r"(^\d+)"
    regex = re.compile(pattern)
    list_with_number = regex.findall(filename)
    return list_with_number[0]


def group_number(group: str) -> int:
    """
    Given a group name, returns it's corresponding number, ie:
    control -> 0
    mania -> 1
    etc...
    """
    groupnames = ['control',
                  'mania',
                  'mixed mania',
                  'mixed depression',
                  'depression',
                  'euthymia']
    return groupnames.index(group)



tasks = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent', 'fcat', 'flib', 'flit']
feature_cols = ['number', 'group', 'group number', 'unique_entries', 'repeat_entries', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']


def features():
    path = "Data/Verbal Tasks 1-3/"
    newpath = "Data/Verbal Tasks features/"
    allfiles = listdir(path)
    csvfiles = sorted([x for x in allfiles if '.csv' in x])

    for task in tasks:
        print(task)
        list_of_features = []
        for file in csvfiles:
            if task in file:
                try:
                    path_name = path + file
                    df = pd.read_csv(path_name)
                    s = df['0']
                    number = get_number(file)
                    group = groups.number_to_group(number)
                    groupnumber = group_number(group)
                    s_vec = gv.vec_series(s)
                    neigh_sim = neighsim.avg_neighbour_sim(s_vec)
                    s_unique = gv.vec_series_unique(s)
                    global_sim = globalsim.global_sim(s_unique)
                    unique_entries = len(s_unique)
                    repeat_entries = s.duplicated().sum()
                    repeat_words = wordrep.word_repeat(s)
                    arr = np.array([number, group, groupnumber, unique_entries, repeat_entries, repeat_words, global_sim, neigh_sim])
                    list_of_features.append(arr)

                except Exception as e:
                    print(e)
                    print('problem with:', file)
        df = pd.DataFrame(list_of_features, columns=feature_cols)
        new_filename = task + '_features.csv'
        df.to_csv(newpath+new_filename, index=False)


if __name__ == '__main__':
    features()





