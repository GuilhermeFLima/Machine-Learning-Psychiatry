# Script to reduce the number of words in each verbal task entry

import pandas as pd
from os import listdir


def word_counter(entry: str) -> int:
    undscr = '_'
    return entry.count(undscr) + 1


def count_filter(n: int, entry: str) -> bool:
    return word_counter(entry) <= n


def entry_reduce(n: int, newpath: str):
    path = "Data/Verbal Tasks redux/"
    allfiles = listdir(path)
    csvfiles = [x for x in allfiles if '.csv' in x]

    for file in csvfiles:
        try:
            filename = path + file
            df = pd.read_csv(filename)
            s = df['0']
            s = s[s.map(lambda x: count_filter(n, x))]
            s = s.reset_index().iloc[:, 1]
            newfilename = newpath + file
            s.to_csv(newfilename)
        except Exception as e:
            print(e)

    return None


if __name__ == '__main__':
    n = 3
    newpath = "Data/Verbal Tasks 1-3/"
    entry_reduce(n=n, newpath=newpath)
