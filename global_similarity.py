import pandas as pd
import fasttext.util
import numpy as np
from scipy import spatial


def cos_sim(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def global_sim(series):
    if len(series) < 2:
        return 0.0
    else:
        sim_list = []
        for (i, vec1) in series.iteritems():
            for (j, vec2) in series.iteritems():
                if i < j:
                    sim = cos_sim(vec1, vec2)
                    sim_list.append(sim)
        return np.mean(sim_list)

