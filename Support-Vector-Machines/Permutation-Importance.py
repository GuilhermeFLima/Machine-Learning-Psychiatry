import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance


def improve_dataframe_display():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.precision', 2)
    return None


def group_select(dataframe, group1, group2):
    mask = (dataframe['group'] == group1) | (dataframe['group'] == group2)
    return dataframe[mask]


def split(dataframe):
    to_drop = ['number', 'group', 'group number']
    df_X = dataframe.drop(to_drop, axis=1)
    df_y = column_or_1d(y=dataframe[['group number']], warn=False)
    return df_X, df_y


features = ['unique_entries', 'repeat_entries', 'repeat_words', 'avg_global_sim', 'avg_neigh_sim']

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
    GSCV_results_file = group1 + '_vs_' + group2 + '_GSCV_results.csv'
    grid_search_results = pd.read_csv(GSCV_results_file)
    importance_mean = pd.DataFrame(columns=['task'] + features)
    importance_std = pd.DataFrame(columns=['task'] + features)
    for (j, (task, file)) in enumerate(zip(tasks, csv_files)):

        # Reading the csv file into a dataframe and selecting the groups
        path = "../Data/Verbal Tasks features/"
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

        # Getting Support-Vector-Machines parameters
        kernel = grid_search_results.loc[j]['kernel']
        C = grid_search_results.loc[j]['C']
        if kernel == 'rbf':
            gamma = grid_search_results.loc[j]['gamma']

        # Building the classifier using parameters
        svc = SVC(kernel=kernel, C=C, gamma=gamma)
        svc.fit(X_train_scaled, y_train)

        # Calculating permutation importance
        perm_importance = permutation_importance(svc, X_test_scaled, y_test, random_state=0)
        importance_mean.loc[j] = [task] + list(perm_importance.importances_mean)
        importance_std.loc[j] = [task] + list(perm_importance.importances_std)

    print(importance_mean)
    importance_mean_file = group1 + '_vs_' + group2 + '_permutation_importance_mean.csv'
    importance_mean.to_csv(importance_mean_file, index=False)
    print('\n')
    print(importance_std)
    importance_std_file = group1 + '_vs_' + group2 + '_permutation_importance_std.csv'
    importance_std.to_csv(importance_std_file, index=False)

