import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import train_test_split
from tasks_features import task_features


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

results = pd.DataFrame(columns=['task', 'training', 'test'])
for (i, (task, cols)) in enumerate(task_features.items()):
    df_X_task = df_X[cols]
    X_train, X_test, y_train, y_test = train_test_split(df_X_task, df_y, random_state=0)
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0)
    rfc.fit(X_train, y_train)
    results.loc[i] = [task, rfc.score(X_train, y_train), rfc.score(X_test, y_test)]

print(results)