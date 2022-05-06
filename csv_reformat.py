# This script reformats the csv files.
# Originally the csv files for each verbal fluency task where not vertical,
# as can be seen in the folder 'Verbal Tasks original'. When transposing them, pandas
# views the entries as column names, and thus adds numbers to repeated entries,
# which is why we need the remove_number function. We also do som cleaning with the
# functions that remove underscores and uppercases.
# The output is saved to 'Verbal Tasks original reformated'.

import pandas as pd
from os import listdir


def remove_number(word: str) -> str:
    L = word.split('.')
    if len(L) == 2:
        if L[1].isnumeric():
            return L[0]
    else:
        return word


def remove_underscore(word: str) -> str:
    if word[0] == '_':
        return word[1:]
    elif word[-1] == '_':
        return word[:-1]
    else:
        return word


def remove_doubleunderscore(word: str) -> str:
    new_word = word.replace('__', '_')
    return new_word


def remove_uppercase(word: str) -> str:
    new_word = word.lower()
    return new_word


def csv_reformat():
    path = "Recycling/Verbal Tasks original/"
    newpath = "Data/Verbal Tasks/"
    allfiles = listdir(path)
    csvfiles = [x for x in allfiles if '.csv' in x]

    for file in csvfiles:
        try:
            filename = path + file
            df = pd.read_csv(filename)
            s = pd.Series(list(df.columns))
            s = s.map(remove_number)
            s = s.map(remove_underscore)
            s = s.map(remove_doubleunderscore)
            s = s.map(remove_uppercase)
            newfilename = newpath + file
            s.to_csv(newfilename)
        except Exception as e:
            print(e)
            print('problem with:', file)

    return None


if __name__ == '__main__':
    csv_reformat()

