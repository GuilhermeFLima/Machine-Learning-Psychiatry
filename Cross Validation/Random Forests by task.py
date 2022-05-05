
# clf = svm.SVC(kernel='linear', C=1, random_state=42)
# scores = cross_val_score(clf, X, y, cv=5)
# scores

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from tasks_features import task_features
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']

df = pd.read_csv("../Data/Verbal Tasks joined features/joined_features.csv")


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

    LOO = LeaveOneOut()
    rfc = RandomForestClassifier(max_depth=None, n_estimators=2000)
    scores = cross_val_score(rfc, df_X_task, df_y, cv=10)

    # svc = grid_search.fit(X_train, y_train)
    # pred_svc = svc.predict(X_test)
    # conf_matrix = confusion_matrix(y_test, pred_svc)
    # F1 = f1_score(y_test, pred_svc)
    # MCC = matthews_corrcoef(y_test, pred_svc)
    # test_score = grid_search.score(X_test, y_test)
    # best_params = dict.fromkeys(['max_depth'])
    # best_params.update(grid_search.best_params_)
    # best_max_depth = best_params['max_depth']
    # best_cv = grid_search.best_score_
    # results.loc[j] = [task, test_score, F1, MCC, best_max_depth, best_cv]
    print(task, np.median(scores))
    #print("Confusion matrix:\n{}".format(conf_matrix))

# desired_width = 320
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns', 10)
# pd.set_option('display.precision', 2)
# print('\n')
# print(results)
# print('Average accuracy: {:.2f}'.format(results['acc'].mean()))
# print('Accuracy SD: {:.2f}'.format(results['acc'].std()))
# print('Max accuracy: {:.2f}'.format(results['acc'].max()))