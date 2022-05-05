import pandas as pd


def features_dataframes(files_list: list) -> list:
    """
    Given the list of csv files, returns the list of dataframes, and removes
    the columns 'number' and 'group', since these will be added later.
    """
    path = "Data/Verbal Tasks features/"
    files = [path + file for file in files_list]
    dataframes = [pd.read_csv(file) for file in files]
    for df in dataframes:
        df.drop(['number', 'group'], axis=1, inplace=True)
    return dataframes


def join_dataframes(dataframes: list):
    """
    Joins the dataframes and renames the columns, since these were double-indexed.
    """
    total_tasks = ['courage', 'debut', 'douleur', 'piscine',
                   'royaume', 'serpent', 'fcat', 'flib', 'flit']
    joined = pd.concat(dataframes, axis=1, keys=total_tasks)
    joined.columns = ['_'.join(col) for col in joined.columns.values]
    return joined


def group_number(group: str) -> int:
    """
    Given a group name, returns it's corresponding number, ie:
    control -> 0
    mania -> 1
    etc...
    """
    groupnames = ['control',
                  'mania',
                  'mixed mania',
                  'mixed depression',
                  'depression',
                  'euthymia']
    return groupnames.index(group)


def number_group_groupnumber():
    """
    Returns a dataframe with the patient number, its group, and the group's number.
    """
    file = "Data/Verbal Tasks features/courage_features.csv"
    df = pd.read_csv(file)
    df = df[['number', 'group']]
    df['group number'] = df['group'].map(group_number)
    return df


if __name__ == '__main__':
    csv_files = ['courage_features.csv',
                 'debut_features.csv',
                 'douleur_features.csv',
                 'piscine_features.csv',
                 'royaume_features.csv',
                 'serpent_features.csv',
                 'fcat_features.csv',
                 'flib_features.csv',
                 'flit_features.csv']
    dfs = features_dataframes(files_list=csv_files)
    joined_tasks = join_dataframes(dataframes=dfs)
    df0 = number_group_groupnumber()
    final_dataframe = pd.concat([df0, joined_tasks], axis=1)
    final_dataframe.to_csv('Data/Verbal Tasks joined features/joined_features2.csv', index=False)


