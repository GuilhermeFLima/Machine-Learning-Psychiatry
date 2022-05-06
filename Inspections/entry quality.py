import pandas as pd
import numpy as np
from os import listdir

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

anchor_list = ['courage', 'debut', 'douleur', 'piscine', 'royaume', 'serpent']
simple_fluence_list = ['fcat', 'flib', 'flit']
tasks = anchor_list + simple_fluence_list

path = "../Data/Verbal Tasks/"
allfiles = listdir(path)

results = pd.DataFrame(columns=['task', 'entries', 'avg', 'std', 'avg_bad', 'std_bad', 'avg_bad_pcnt', 'total_bad'])
for (j, task) in enumerate(tasks):
    bad_entries = []
    bad_percents = []
    entries = []
    for file in allfiles:
        if task in file:
            df = pd.read_csv(path + file)
            i = 0
            for index, word in df['0'].items():
                if 0 < word.count('_'):
                    i += 1
            bad_entries.append(i)
            total_entries = len(df)
            entries.append(total_entries)
            if total_entries == 0:
                bad_percent = 0
            else:
                bad_percent = i/total_entries
            bad_percents.append(bad_percent)

    sum_entries = sum(entries)
    avg_entries = np.mean(entries)
    std_entries = np.std(entries)

    sum_bad_entries = sum(bad_entries)
    avg_bad_entries = np.mean(bad_entries)
    std_bad_entries = np.std(bad_entries)

    avg_bad_percent = np.mean(bad_percents)
    results.loc[j] = [task, sum_entries, avg_entries, std_entries, avg_bad_entries, std_bad_entries, avg_bad_percent, sum_bad_entries]

print(results.head(9))


