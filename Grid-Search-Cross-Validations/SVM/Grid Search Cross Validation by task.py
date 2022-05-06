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

df = pd.read_csv("../../Recycling/Verbal Tasks joined features/joined_features.csv")


group1 = 'mania'
group2 = 'depression'
df_sub = group_select(df, group1, group2)
to_drop = ['number', 'group', 'group number']
df_X = df_sub.drop(to_drop, axis=1)
df_y = column_or_1d(y=df_sub[['group number']], warn=False)

print('Classification of {} vs {} using Support Vector Machines:'.format(group1, group2))

results = pd.DataFrame(columns=['task', 'acc', 'F1', 'MCC', 'kernel', 'C', 'gamma', 'best cv'])
for (j, (task, cols)) in enumerate(task_features.items()):
    df_X_task = df_X[cols]
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
    test_score = grid_search.score(X_test_scaled, y_test)
    best_params = dict.fromkeys(['C', 'gamma', 'kernel'])
    best_params.update(grid_search.best_params_)
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    kernel = best_params['kernel']
    best_cv = grid_search.best_score_
    results.loc[j] = [task, test_score, F1, MCC, kernel, best_C, best_gamma, best_cv]
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
results.to_csv('GSCV_results.csv')

# task   acc    F1   MCC kernel         C     gamma  best cv
# 0  courage  0.82  0.88  0.54    rbf  1.00e+00  3.16e+01     0.71
# 1    debut  0.73  0.84  0.00    rbf  1.00e-03  1.00e-03     0.55
# 2  douleur  0.82  0.88  0.54    rbf  3.16e+02  1.00e+00     0.68
# 3  piscine  0.55  0.67 -0.04    rbf  3.16e+02  1.00e+00     0.65
# 4  royaume  0.55  0.67 -0.04    rbf  3.16e+00  3.16e+01     0.61
# 5  serpent  0.73  0.80  0.39    rbf  3.16e+02  1.00e+00     0.65
# 6     fcat  0.82  0.89  0.52    rbf  1.00e+00  1.00e+02     0.61
# 7     flib  0.82  0.89  0.52    rbf  1.00e+00  1.00e+02     0.61
# 8     flit  0.82  0.89  0.52    rbf  1.00e+00  1.00e+02     0.61
# Average accuracy: 0.74
# Accuracy SD: 0.12
# Max accuracy: 0.82