import pandas as pd

anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']

filename1 = "Data/Verbal Tasks features/courage_features.csv"
filename2 = "Data/Verbal Tasks features/debut_features.csv"
filename3 = "Data/Verbal Tasks features/douleur_features.csv"
filename4 = "Data/Verbal Tasks features/piscine_features.csv"
filename5 = "Data/Verbal Tasks features/royaume_features.csv"
filename6 = "Data/Verbal Tasks features/serpent_features.csv"

df1 = pd.read_csv(filename1)
df1.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)
df2 = pd.read_csv(filename2)
df2.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)
df3 = pd.read_csv(filename3)
df3.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)
df4 = pd.read_csv(filename4)
df4.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)
df5 = pd.read_csv(filename5)
df5.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)
df6 = pd.read_csv(filename6)
df6.drop(['Unnamed: 0', 'number', 'group'], axis=1, inplace=True)

df0 = pd.concat([df1, df2, df3, df4, df5, df6], axis=1, keys=anchor_list)
df0.columns = ['_'.join(col) for col in df0.columns.values]

df_num_group = pd.read_csv(filename1)
df_num_group = df_num_group[['number', 'group']]
groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
df_num_group['group number'] = df_num_group['group'].map(lambda n: groupnames.index(n))
df = pd.concat([df_num_group, df0], axis=1)
df.to_csv('Data/Verbal Tasks joined features/joined_features.csv')