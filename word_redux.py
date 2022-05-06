# This script just mostly cleans the verbal fluency tasks by:
# i) removing connectives (pronouns + prepositions)
# ii) removing underscores at the _beginning and end_ of an entry.

from Inspections.word_inspect import _connectives_, connectives_
import pandas as pd
from os import listdir


def remove_connective_mid(word: str) -> str:
    new_word = word
    for c in _connectives_:
        new_word = new_word.replace(c, '_')
    return new_word


def remove_connective_start(word: str) -> str:
    new_word = word
    for c in connectives_:
        n = len(c)
        if len(new_word) > n:
            if c == new_word[:n]:
                new_word = new_word[n:]
    return new_word


def remove_underscores(word: str) -> str:
    if '_' == word[0]:
        new_word = word[1:]
    else:
        new_word = word

    if '_' == new_word[-1]:
        new_word = new_word[:-1]

    return new_word


def word_redux():
    path = "Data/Verbal Tasks/"
    newpath = "Data/Verbal Tasks redux/"
    allfiles = listdir(path)
    csvfiles = [x for x in allfiles if '.csv' in x]

    for file in csvfiles:
        try:
            filename = path + file
            df = pd.read_csv(filename)
            s = df['0']
            s = s.map(remove_connective_mid)
            s = s.map(remove_connective_start)
            s = s.map(remove_connective_mid)
            s = s.map(remove_underscores)
            newfilename = newpath + file
            s.to_csv(newfilename)
        except:
            print('problem with:', file)

    return None


if __name__ == '__main__':
    word_redux()




