import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
anchor_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']
simple_fluence_list = ['fcat', 'flib', 'flit']
simple_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']


df = pd.read_csv("../../Recycling/Verbal Tasks joined features/joined_features.csv")
group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
features_to_drop = [task + '_avg_anch_sim' for task in anchor_list]
to_drop = ['Unnamed: 0', 'number', 'group', 'group number'] + features_to_drop
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_range = [10**x for x in np.arange(-3, 3, 0.5)]
param_grid = [{'kernel': ["rbf"], 'C': param_range, 'gamma': param_range},
              {'kernel': ["linear"], 'C': param_range}
              ]

LOO = LeaveOneOut()
grid_search = GridSearchCV(SVC(), param_grid, cv=LOO, return_train_score=True)

grid_search.fit(X_train_scaled, y_train)
print('Classification of depression vs mania using Support Vector Machines:')
print("Test set score: {:.2f}".format(grid_search.score(X_test_scaled, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# results = pd.DataFrame(grid_search.cv_results_)
# desired_width = 320
# pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_columns', 10)
# print(results[['param_C', 'param_gamma',
#                'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score'
#                ]])
