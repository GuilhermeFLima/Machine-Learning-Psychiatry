import pandas as pd
import matplotlib.pyplot as plt


filename = "Data/Verbal Tasks features/douleur_features.csv"
df = pd.read_csv(filename)
df.drop(['number', 'Unnamed: 0'], axis=1, inplace=True)
df = df[(df['group'] == 'mania') | (df['group'] == 'depression')]
df1 = df[['avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']]
df2 = df[['unique_words', 'repeat_words']]

groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
s_groups = df['group']
s_groups = s_groups.map(lambda n: groupnames.index(n))

pd.plotting.scatter_matrix(df2, c=s_groups, marker='o')
plt.legend()
plt.show()