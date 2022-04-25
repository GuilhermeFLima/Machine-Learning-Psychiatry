import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import column_or_1d
import group_merge as gm
from sklearn.preprocessing import MinMaxScaler


def group_select(dataframe, group1, group2, group3):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2) | (dataframe['group'] == group3)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']

df = pd.read_csv("../Data/Verbal Tasks joined features/joined_features.csv")

df = gm.group_merge(df, 'mania', 'mixed mania')
df = gm.group_merge(df, 'depression', 'mixed depression')
group1 = gm.name_merge('mania', 'mixed mania')
group2 = gm.name_merge('depression', 'mixed depression')
group3 = group2
df_sub = group_select(df, group1, group2, group3)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = LogisticRegression(solver='liblinear', penalty='l1', C=10).fit(X_train_scaled, y_train)
print("Training set score: {:.2f}".format(log_reg.score(X_train_scaled, y_train)))
print("Test set score: {:.2f}".format(log_reg.score(X_test_scaled, y_test)))
features = df_X.columns
coeficients = log_reg.coef_[0]
for (f, c) in zip(features, coeficients):
    print(f, c)

