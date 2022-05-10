import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 2)


means = pd.read_csv('../Support-Vector-Machines/permutation_importance_mean.csv')
means = means.set_index('task')
stds = pd.read_csv('../Support-Vector-Machines/permutation_importance_std.csv')
stds = stds.set_index('task')


task = 'serpent'
fig, ax = plt.subplots()
features = means.columns
y_pos = np.arange(len(features))
X = means.loc[task]
Xerr = stds.loc[task]
ax.barh(y_pos, X, xerr=Xerr)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
plt.tight_layout(True)
plt.show()

