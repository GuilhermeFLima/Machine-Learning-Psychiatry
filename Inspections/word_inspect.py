import pandas as pd
from os import listdir


prepositions = ['a', 'à', 'de', 'des', 'du', 'chez', 'en', 'sur', 'sous', 'au', 'dans', 'par', 'chez', 'et', 'avec', 'aux', 'pour']
pronouns = ['ça',
            'ce',
            'elle',
            'elles',
            'en',
            'eux',
            'il',
            'ils',
            'je',
            'la',
            'le',
            'les',
            'lui',
            'ma',
            'me',
            'mes',
            'mon',
            'moi',
            'nos',
            'nous',
            'on',
            'qui',
            'sa',
            'se',
            'ses',
            'soi',
            'te',
            'toi',
            'tu',
            'un',
            'une',
            'vous',
            'y']


connectives = prepositions + pronouns
apost = ['l\'', 'qu\'', 'd\'', 'm\'', 'c\'', 's\'', 'j\'']
_connectives_ = ['_' + c + '_' for c in connectives] + ['_' + c for c in apost]
connectives_ = [c + '_' for c in connectives] + apost

if __name__ == "__main__":

    path = "Data/Verbal Tasks redux/"
    allfiles = listdir(path)
    csvfiles = [x for x in allfiles if '.csv' in x]

    i = 0
    for file in csvfiles:
        df = pd.read_csv(path+file)
        for index, word in df['0'].items():
            if 1 < word.count('_') < 3:
                i += 1
                print(i, word, file)

