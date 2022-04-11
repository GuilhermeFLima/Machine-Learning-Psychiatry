import get_vec as gv
import anchor_similarity as anchsim
import neighbour_similarity as neighsim
import global_similarity as globsim
import pandas as pd

filename = "Data/Verbal Tasks 1-3/01 douleur words.csv"
df = pd.read_csv(filename)
s = df['0']
s_vec = gv.vec_series(s)
neigh_sim = neighsim.avg_neighbour_sim(s_vec)
anch_word = anchsim.get_anchor(filename)
anch_vec = gv.get_vec(anch_word)
anch_sim = anchsim.avg_anchor_sim(anchor_vec=anch_vec, series=s_vec)
s_unique = gv.vec_series_unique(s)
glob_sim = globsim.global_sim(s_unique)
print('Average neighbour similarity: ', neigh_sim)
print('Average anchor similarity: ', anch_sim)
print('Average global similarity: ', glob_sim)
print('Unique words: ', len(s_unique))
print('Repeats:', s.duplicated().sum())
print('Total words: ', len(s))