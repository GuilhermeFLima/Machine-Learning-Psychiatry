import pandas as pd
from os import listdir

#vt = ['courage', 'debut', 'douleur', 'fcat', 'flit', 'piscines', 'royaume', 'serpent']

path = "Data/Verbal Tasks/"
newpath = "Data/Verbal Tasks reformated/"
allfiles = listdir(path)
csvfiles = [x for x in allfiles if '.csv' in x ]


for file in csvfiles[5:]:
    try:
        filename = path + file
        df = pd.read_csv(filename)
        s = pd.Series(list(df.columns))
        newfilename = newpath + file
        s.to_csv(newfilename)
    except:
        print('problem with:', file)

