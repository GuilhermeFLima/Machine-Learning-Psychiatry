import pandas as pd
import fasttext.util
import numpy as np
from scipy import spatial


def cos_sim(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def avg_anchor_sim(anchor_vec, series):
    similarities = []
    for (i, vec) in series.iteritems():
        sim = cos_sim(anchor_vec, vec)
        print(sim)
        similarities.append(sim)
    return np.mean(similarities)


anchor_list =['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']


def get_anchor(filename: str) -> str:
    for word in anchor_list:
        if word in filename:
            return word


if __name__ == '__main__':
    pass
