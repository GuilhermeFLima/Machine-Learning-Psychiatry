import pandas as pd
import fasttext.util
import numpy as np
from scipy import spatial


def cos_sim(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def get_vec(phrase: str):
    word_list = phrase.split('_')
    vec_list = [ft.get_word_vector(word) for word in word_list]
    final_vec = np.sum(vec_list, axis=0)
    return final_vec


def avg_neighbour_sim(series):
    similarities = []
    v0 = None
    for (i, phrase) in series.iteritems():
        v1 = get_vec(phrase)
        if not(v0 is None):
            sim = cos_sim(v0, v1)
            similarities.append(sim)
            v0 = v1
        else:
            v0 = v1
    return np.mean(similarities)


if __name__ == '__main__':
    # fasttext.util.download_model('fr', if_exists='ignore')
    ft = fasttext.load_model('cc.fr.300.bin')


