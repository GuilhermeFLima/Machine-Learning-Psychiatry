import pandas as pd
import fasttext.util
import numpy as np
from scipy import spatial

#fasttext.util.download_model('fr', if_exists='ignore')
ft = fasttext.load_model('cc.fr.300.bin')


def cos_sim(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def get_vec(phrase: str):
    word_list = phrase.split('_')
    vec_list = [ft.get_word_vector(word) for word in word_list]
    final_vec = np.sum(vec_list, axis=0)
    return final_vec


def avg_anchor_sim(anchor_word, series):
    similarities = []
    anchor_vec = ft.get_word_vector(anchor_word)
    for (i, phrase) in series.iteritems():
        vec = get_vec(phrase)
        sim = cos_sim(anchor_vec, vec)
        print(sim)
        similarities.append(sim)
    return np.mean(similarities)

anchor = 'chien'
words = ['chat', 'chien', 'chien_noir', 'maison', 'roi']
s = pd.Series(words)
print(avg_anchor_sim(anchor_word=anchor, series=s))
