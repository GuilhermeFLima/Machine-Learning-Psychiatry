import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
anchor_cols = ['unique_words', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']
simple_fluence_list = ['fcat', 'flib', 'flit']
simple_cols = ['unique_words', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']

df = pd.read_csv("../Verbal Tasks joined features/joined_features.csv")
group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
features_to_drop = [task + '_repeat_words' for task in anchor_list + simple_fluence_list]
to_drop = ['Unnamed: 0', 'number', 'group', 'group number'] + features_to_drop
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=3)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


svc = SVC(kernel='rbf', C=1, gamma=0.1).fit(X_train_scaled, y_train)
print("Training set score: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}".format(svc.score(X_test_scaled, y_test)))
#print("Number of features used:", np.sum(logreg.coef_ != 0))
