import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from tasks_features import task_features
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 2)


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
anchor_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']
anchor_cols = ['avg_anch_sim']
simple_fluence_list = ['fcat', 'flib', 'flit']
simple_cols = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']

df = pd.read_csv("../../Recycling/Verbal Tasks joined features/joined_features.csv")


group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

# print('Classification of {} vs {} using Support Vector Machines:'.format(group1, group2))

accuracy_comparison = pd.DataFrame(columns=['removed', 'avg', 'max', 'std'])
for (j, feature) in enumerate(anchor_cols):
    print('')
    print('Removing', feature)
    results = pd.DataFrame(columns=['task', 'acc', 'F1', 'MCC', 'kernel', 'C', 'gamma', 'best cv'])
    for (i, (task, cols)) in enumerate(task_features.items()):
        df_X_task = df_X[cols]
        feature_to_drop = task + '_' + feature
        if feature_to_drop in df_X_task.columns:
            df_X_task = df_X_task.drop([feature_to_drop], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df_X_task, df_y, random_state=0)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        param_range = [10**x for x in np.arange(-3, 3, 0.5)]
        param_grid = [{'kernel': ["rbf"], 'C': param_range, 'gamma': param_range},
                      {'kernel': ["linear"], 'C': param_range}
                      ]

        LOO = LeaveOneOut()
        grid_search = GridSearchCV(SVC(), param_grid, cv=10, return_train_score=True)

        svc = grid_search.fit(X_train_scaled, y_train)
        pred_svc = svc.predict(X_test_scaled)
        conf_matrix = confusion_matrix(y_test, pred_svc)
        F1 = f1_score(y_test, pred_svc)
        MCC = matthews_corrcoef(y_test, pred_svc)
        test_score = grid_search.score(X_test_scaled, y_test)
        best_params = dict.fromkeys(['C', 'gamma', 'kernel'])
        best_params.update(grid_search.best_params_)
        best_C = best_params['C']
        best_gamma = best_params['gamma']
        kernel = best_params['kernel']
        best_cv = grid_search.best_score_
        results.loc[i] = [task, test_score, F1, MCC, kernel, best_C, best_gamma, best_cv]
        print(task, end=' ')
        # print("Confusion matrix:\n{}".format(conf_matrix))
    print('\n')
    print(results)
    acc = results['acc'].mean()
    max = results['acc'].max()
    std = results['acc'].std()
    accuracy_comparison.loc[j] = [feature, acc, max, std]

print('\n')
print(accuracy_comparison)




# only linear kernels
# Removing avg_anch_sim
# courage debut douleur piscine royaume serpent fcat flib flit
#
#       task   acc    F1   MCC  kernel         C gamma  best cv
# 0  courage  0.91  0.93  0.81  linear  3.16e+01  None     0.62
# 1    debut  0.73  0.84  0.00  linear  1.00e-03  None     0.55
# 2  douleur  0.82  0.86  0.67  linear  1.00e+02  None     0.57
# 3  piscine  0.45  0.50  0.04  linear  1.00e+01  None     0.58
# 4  royaume  0.64  0.75  0.08  linear  1.00e+02  None     0.68
# 5  serpent  0.73  0.84  0.00  linear  1.00e-03  None     0.55
# 6     fcat  0.73  0.82  0.24  linear  1.00e+02  None     0.81
# 7     flib  0.55  0.55  0.38  linear  3.16e+00  None     0.65
# 8     flit  0.45  0.57 -0.15  linear  1.00e+01  None     0.61
#
#
#         removed   avg   max   std
# 0  avg_anch_sim  0.67  0.91  0.16