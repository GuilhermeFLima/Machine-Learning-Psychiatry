import pandas as pd
from os import listdir

path = "../Data/Verbal Tasks features/"
files = listdir(path)
csvfiles = [file for file in files if '.csv' in file]

for file in csvfiles:
    df = pd.read_csv(path + file)
    df = df[df['group'] == 'mania']
    group_numbers = set(df['group number'].tolist())
    print(group_numbers)