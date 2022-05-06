import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
import group_merge as gm
from sklearn.preprocessing import MinMaxScaler


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']

df = pd.read_csv("../../Recycling/Verbal Tasks joined features/joined_features.csv")

df = gm.group_merge(df, 'mania', 'mixed mania')
df = gm.group_merge(df, 'depression', 'mixed depression')
group1 = gm.name_merge('mania', 'mixed mania')
group2 = gm.name_merge('depression', 'mixed depression')

df_sub = group_select(df, group1, group2)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_range = [10**x for x in np.arange(-3, 3, 0.5)]
param_grid = [{'kernel': ["rbf"], 'C': param_range, 'gamma': param_range},
              {'kernel': ["linear"], 'C': param_range}
              ]

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_scaled, y_train)
print('Classification of depression & mixed depression vs mania & mixed mania using Support Vector Machines:')
print("Test set score: {:.2f}".format(grid_search.score(X_test_scaled, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))