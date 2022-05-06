import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def improve_dataframe_display():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.precision', 2)
    return None


def group_select(dataframe, group_1, group_2):
    mask = (dataframe['group'] == group_1) | (dataframe['group'] == group_2)
    return dataframe[mask]


def split(dataframe):
    to_drop = ['number', 'group', 'group number']
    df_X = dataframe.drop(to_drop, axis=1)
    df_y = column_or_1d(y=dataframe[['group number']], warn=False)
    return df_X, df_y


csv_files = ['courage_features.csv',
             'debut_features.csv',
             'douleur_features.csv',
             'piscine_features.csv',
             'royaume_features.csv',
             'serpent_features.csv',
             'fcat_features.csv',
             'flib_features.csv',
             'flit_features.csv']

tasks = ['courage',
         'debut',
         'douleur',
         'piscine',
         'royaume',
         'serpent',
         'fcat',
         'flib',
         'flit']


if __name__ == '__main__':
    improve_dataframe_display()

    group1 = 'mania'
    group2 = 'depression'
    print('Classification of {} vs {} using Support Vector Machines:'.format(group1, group2))

    results = pd.DataFrame(columns=['task', 'acc', 'F1', 'MCC', 'kernel', 'C', 'gamma', 'best cv'])
    for (j, (task, file)) in enumerate(zip(tasks, csv_files)):

        # Reading the csv file into a dataframe and selecting the groups
        path = "../../Data/Verbal Tasks features/"
        df = pd.read_csv(path + file)
        df = group_select(df, group1, group2)

        # Splitting into features X, and targets y
        df_X, df_y = split(df)

        # Splitting into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

        # Scaling for SMVs
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Defining the range of the parameters that will be tuned
        param_range = [10**x for x in np.arange(-3, 3, 0.5)]
        param_grid = [{'kernel': ["rbf"], 'C': param_range, 'gamma': param_range},
                      {'kernel': ["linear"], 'C': param_range}
                      ]

        # Using the Leave-One-Out cross-validator
        LOO = LeaveOneOut()

        # Initialize GridSearchCV class
        grid_search = GridSearchCV(SVC(), param_grid, cv=10, return_train_score=True)
        # Fit to training data
        svc = grid_search.fit(X_train_scaled, y_train)

        # Extracting parameters and kernel and best scores
        best_params = dict.fromkeys(['C', 'gamma', 'kernel'])
        best_params.update(grid_search.best_params_)
        best_C = best_params['C']
        best_gamma = best_params['gamma']
        kernel = best_params['kernel']
        best_cv = grid_search.best_score_

        # Predictions
        pred_svc = svc.predict(X_test_scaled)
        # Extracting accuracy, F1, MCC from predictions
        conf_matrix = confusion_matrix(y_test, pred_svc)
        F1 = f1_score(y_test, pred_svc)
        MCC = matthews_corrcoef(y_test, pred_svc)
        test_score = grid_search.score(X_test_scaled, y_test)

        # Appending to results dataframe
        results.loc[j] = [task, test_score, F1, MCC, kernel, best_C, best_gamma, best_cv]
        print(task, end=' ')

    print('\n')
    print(results)
    print('Average accuracy: {:.2f}'.format(results['acc'].mean()))
    print('Accuracy SD: {:.2f}'.format(results['acc'].std()))
    print('Max accuracy: {:.2f}'.format(results['acc'].max()))
    results.to_csv('GSCV_results.csv', index=False)
