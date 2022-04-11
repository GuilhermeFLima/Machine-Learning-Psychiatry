import pandas as pd
import fasttext.util
import numpy as np
from scipy import spatial


def cos_sim(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def avg_neighbour_sim(series):
    similarities = []
    v0 = None
    for (i, v1) in series.iteritems():
        if not(v0 is None):
            sim = cos_sim(v0, v1)
            similarities.append(sim)
            v0 = v1
        else:
            v0 = v1
    return np.mean(similarities)


if __name__ == '__main__':
    pass


