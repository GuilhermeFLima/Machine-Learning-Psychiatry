# merges groups and removes others.
# Mostly interested in merging mania and mixed mania, and
# depression and mixed depression.

import pandas as pd


groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']


def group_swap(group0, group1, group2):
    if group0 == group1:
        return group2
    else:
        return group0


def group_num_merge(dataframe, group_num1: int, group_num2: int):
    df = dataframe
    df['group number'] = df['group number'].map(lambda group_num: group_swap(group_num, group_num1, group_num2))
    return df


def name_merge(name1: str, name2: str):
    return name1 + ' & ' + name2


def group_name_merge(dataframe, group_name1: str, group_name2: str):
    df = dataframe
    merged_name = name_merge(group_name1, group_name2)
    df['group'] = df['group'].map(lambda group_name: group_swap(group_name, group_name1, merged_name))
    df['group'] = df['group'].map(lambda group_name: group_swap(group_name, group_name2, merged_name))
    return df


def group_merge(dataframe, group_name1: str, group_name2: str):
    group_num1 = groupnames.index(group_name1)
    group_num2 = groupnames.index(group_name2)
    df = dataframe
    df = group_num_merge(df, group_num1, group_num2)
    df = group_name_merge(df, group_name1, group_name2)
    return df


if __name__ == "__main__":
    testdf = pd.read_csv("Recycling/Verbal Tasks joined features/joined_features.csv")
    print(testdf.head(10))
    group1 = 'mania'
    group2 = 'mixed mania'
    df_merged = group_merge(testdf, group1, group2)
    print(df_merged[['group', 'group number']].head(20))



