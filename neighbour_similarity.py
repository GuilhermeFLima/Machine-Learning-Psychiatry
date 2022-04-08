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


dog_vec = ft.get_word_vector('chien')
cat_vec = ft.get_word_vector('chat')
white_vec = ft.get_word_vector('blanc')
black_vec = ft.get_word_vector('noir')
big_vec = ft.get_word_vector('gros')
small_vec = ft.get_word_vector('petit')

vec_dict = {'chien': dog_vec,
            'chat': cat_vec,
            'blanc': white_vec,
            'noir': black_vec,
            'gros': big_vec,
            'petit': small_vec}

words = vec_dict.keys()
doubles = [(a, b)
           for (i, a) in enumerate(words)
           for (j, b) in enumerate(words)
           if i < j]

triples = [(a, b, c)
           for (i, a) in enumerate(words)
           for (j, b) in enumerate(words)
           for (k, c) in enumerate(words)
           if (i < j) and (j < k)]

double_phrases = [a + '_' + b for (a, b) in doubles]
triple_phrases = [a + '_' + b + '_' + c for (a, b, c) in triples]
double_series = pd.Series(double_phrases)
print(avg_neighbour_sim(double_series))

# for (v, p) in zip(triples, triple_phrases):
#     v1 = vec_dict[v[0]]
#     v2 = vec_dict[v[1]]
#     v3 = vec_dict[v[2]]
#     w = v1 + v2 + v3
#     z = get_vec(p)
#     print(v, p)
#     print(np.array_equal(w, z))

# phrases1 = ['chien_blanc', 'chat_blanc', 'chien_noir', 'chat_noir',
#             'petit_chien', 'petit_chat', 'gros_chien', 'gros_chat']
#
# phrases2 = ['petit_chien_blanc']