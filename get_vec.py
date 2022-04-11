import fasttext.util
import numpy as np
import pandas as pd
from scipy import spatial


# fasttext.util.download_model('fr', if_exists='ignore')
print('Loading French FastText model...')
ft = fasttext.load_model('cc.fr.300.bin')
print('Done.')


def get_vec(phrase: str):
    word_list = phrase.split('_')
    vec_list = [ft.get_word_vector(word) for word in word_list]
    final_vec = np.sum(vec_list, axis=0)
    return final_vec


def vec_series(series):
    s = series.map(get_vec)
    return s


def vec_series_unique(series):
    unique = pd.Series(series.unique())
    return vec_series(unique)




