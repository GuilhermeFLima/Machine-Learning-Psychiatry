import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from tasks_features import task_features
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']

df = pd.read_csv("../../Data/Verbal Tasks joined features/joined_features.csv")


group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
to_drop = ['Unnamed: 0', 'number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

print('Classification of {} vs {} using Random Forests:'.format(group1, group2))

results = pd.DataFrame(columns=['task', 'acc', 'F1', 'MCC', 'max_depth', 'best cv'])
for (j, (task, cols)) in enumerate(task_features.items()):
    df_X_task = df_X[cols]
    X_train, X_test, y_train, y_test = train_test_split(df_X_task, df_y, random_state=0)

    param_range = [5, 8, 10, 13, 15]
    param_grid = [{'max_depth': param_range}
                  ]

    LOO = LeaveOneOut()
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators=1000), param_grid, cv=LOO, return_train_score=True)

    svc = grid_search.fit(X_train, y_train)
    pred_svc = svc.predict(X_test)
    conf_matrix = confusion_matrix(y_test, pred_svc)
    F1 = f1_score(y_test, pred_svc)
    MCC = matthews_corrcoef(y_test, pred_svc)
    test_score = grid_search.score(X_test, y_test)
    best_params = dict.fromkeys(['max_depth'])
    best_params.update(grid_search.best_params_)
    best_max_depth = best_params['max_depth']
    best_cv = grid_search.best_score_
    results.loc[j] = [task, test_score, F1, MCC, best_max_depth, best_cv]
    print(task, end=' ')
    #print("Confusion matrix:\n{}".format(conf_matrix))

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 2)
print('\n')
print(results)
print('Average accuracy: {:.2f}'.format(results['acc'].mean()))
print('Accuracy SD: {:.2f}'.format(results['acc'].std()))
print('Max accuracy: {:.2f}'.format(results['acc'].max()))

# 1000 trees used:
#       task   acc    F1   MCC  max_depth  best cv
# 0  courage  0.91  0.93  0.81          5     0.55
# 1    debut  0.36  0.46 -0.26          5     0.52
# 2  douleur  0.73  0.80  0.39          5     0.55
# 3  piscine  0.82  0.88  0.54          8     0.45
# 4  royaume  0.27  0.33 -0.39          8     0.65
# 5  serpent  0.73  0.80  0.39          8     0.48
# 6     fcat  0.64  0.71  0.26          8     0.48
# 7     flib  0.64  0.71  0.26          5     0.48
# 8     flit  0.64  0.71  0.26         15     0.48
# Average accuracy: 0.64
# Accuracy SD: 0.20
# Max accuracy: 0.91