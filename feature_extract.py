import get_vec as gv
import anchor_similarity as anchsim
import neighbour_similarity as neighsim
import global_similarity as globalsim
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


anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
cols = ['number', 'group', 'unique_words', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']


def features():
    path = "Data/Verbal Tasks 1-3/"
    newpath = "Data/Verbal Tasks features/"
    allfiles = listdir(path)
    csvfiles = sorted([x for x in allfiles if '.csv' in x])

    for anchor in anchor_list:
        print(anchor)
        list_of_features = []
        for file in csvfiles:
            if anchor in file:
                try:
                    path_name = path + file
                    df = pd.read_csv(path_name)
                    s = df['0']
                    number = get_number(file)
                    group = groups.number_to_group(number)
                    s_vec = gv.vec_series(s)
                    neigh_sim = neighsim.avg_neighbour_sim(s_vec)
                    anch_word = anchsim.get_anchor(path_name)
                    anch_vec = gv.get_vec(anch_word)
                    anch_sim = anchsim.avg_anchor_sim(anchor_vec=anch_vec, series=s_vec)
                    s_unique = gv.vec_series_unique(s)
                    global_sim = globalsim.global_sim(s_unique)
                    unique_words = len(s_unique)
                    repeat_words = s.duplicated().sum()
                    arr = np.array([number, group, unique_words, repeat_words, anch_sim, global_sim, neigh_sim])
                    list_of_features.append(arr)

                except Exception as e:
                    print(e)
                    print('problem with:', file)
        df = pd.DataFrame(list_of_features, columns=cols)
        new_filename = anchor + '_features.csv'
        df.to_csv(newpath+new_filename)


if __name__ == '__main__':
    features()
