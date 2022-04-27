import matplotlib.pyplot as plt
import pandas as pd
from os import listdir


path = "../Data/Verbal Tasks features/"
allfiles = listdir(path)
fig, axs = plt.subplots(nrows=3, ncols=3)
for (file, ax) in zip(allfiles, axs.flat):
    df = pd.read_csv(path+file)
    g1 = (df['group'] == 'mania')
    g2 = (df['group'] == 'depression')
    X = df['repeat_entries']
    Y = df['repeat_words']
    ax.scatter(X[g1], Y[g1], color='r', alpha=0.7)
    ax.scatter(X[g2], Y[g2], color='b', alpha=0.7)
    ax.set_title(file)

plt.show()