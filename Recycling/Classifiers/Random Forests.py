import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']

df = pd.read_csv("../Verbal Tasks joined features/joined_features.csv")
group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

n_est = 1000
m_dep = None

rfc = RandomForestClassifier(n_estimators=n_est, max_depth=m_dep, random_state=0)
rfc.fit(X_train, y_train)
print("train: {:.2f}, test: {:.2f}".format(rfc.score(X_train, y_train), rfc.score(X_test, y_test)))


# importances = rfc.feature_importances_
# feature_names = df_X.columns
# std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=feature_names)
#print(forest_importances.sort_values(ascending=False))