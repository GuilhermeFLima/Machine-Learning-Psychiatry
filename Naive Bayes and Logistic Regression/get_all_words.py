import pandas as pd
from os import listdir


def get_words(path_to_vf_folder, vf_type):
    all_files = listdir(path_to_vf_folder)
    vf_files = [x for x in all_files if vf_type in x]
    all_words = []
    for file in vf_files:
        df = pd.read_csv(path_to_vf_folder + file)
        s = df['0']
        l = s.tolist()
        all_words += l

    all_words = pd.Series(list(set(all_words)))
    all_words = all_words.sort_values()
    all_words = all_words.reset_index(drop=True)
    return all_words


def get_words_all_vf(path_to_vf_folder):
    vf_list = ['courage', 'debut', 'douleur', 'fcat', 'flib', 'flit', 'piscine', 'royaume', 'serpent']
    for vf in vf_list:
        df = get_words(path_to_vf_folder=path_to_vf_folder, vf_type=vf)
        save_folder = 'All Words/'
        df.to_csv(save_folder + vf + '.csv', index=False)
    return None


if __name__ == '__main__':
    path = '../Data/Verbal Tasks redux split/'
    get_words_all_vf(path_to_vf_folder=path)







