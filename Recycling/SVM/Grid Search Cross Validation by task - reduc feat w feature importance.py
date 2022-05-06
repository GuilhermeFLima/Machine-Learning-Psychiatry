import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from tasks_features import task_features
import matplotlib.pyplot as plt
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
to_drop = ['number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

# print('Classification of {} vs {} using Support Vector Machines:'.format(group1, group2))

accuracy_comparison = pd.DataFrame(columns=['removed', 'avg', 'max', 'std'])
for (j, feature) in enumerate(anchor_cols):
    print('')
    print('Removing', feature)
    results = pd.DataFrame(columns=['task'] + simple_cols)
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
        best_coef = svc.best_estimator_.coef_
        results.loc[i] = [task] + list(best_coef[0])
        print(task)
        # print("Confusion matrix:\n{}".format(conf_matrix))
    print('\n')
    print(results)

df = results.set_index('task')
df = df.iloc[0:6]
plt.imshow(df, cmap='RdYlBu')
plt.colorbar()
plt.xticks(range(len(df.columns)), df, rotation='vertical')
plt.yticks(range(len(df.index)), df.index)
plt.show()