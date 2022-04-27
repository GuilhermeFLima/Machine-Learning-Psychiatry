import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from tasks_features import task_features
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
anchor_cols = ['unique_words', 'repeat_words', 'avg_anch_sim', 'avg_global_sim', 'avg_neigh_sim']
simple_fluence_list = ['fcat', 'flib', 'flit']
simple_cols = ['unique_words', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']

df = pd.read_csv("../Data/Verbal Tasks joined features/joined_features.csv")


group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

print('Classification of {} vs {} using Support Vector Machines:'.format(group1, group2))

for (task, cols) in task_features.items():
    df_X_task = df_X[cols]
    feature_to_drop = [task + '_repeat_words']
    df_X_task = df_X_task.drop(feature_to_drop, axis=1)

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
    grid_search = GridSearchCV(SVC(), param_grid, cv=LOO, return_train_score=True)

    svc = grid_search.fit(X_train_scaled, y_train)
    pred_svc = svc.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, pred_svc)
    F1 = f1_score(y_test, pred_svc)
    MCC = matthews_corrcoef(y_test, pred_svc)


    # print('Task: ', task)
    # print("Test set score: {:.2f}".format(grid_search.score(X_test_scaled, y_test)))
    # print("F1 score: {:.2f}".format(F1))
    # print("MCC score: {:.2f}".format(MCC))
    # print("Best parameters: {}".format(grid_search.best_params_))
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    #print("Confusion matrix:\n{}".format(conf_matrix))

