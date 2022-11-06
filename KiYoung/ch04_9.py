import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (10, 5)})

df = sns.load_dataset('flights')
print(df.head())
print(df.describe())
print(df.isnull().sum())

pivot_data = df.pivot('month', 'year', 'passengers')
print(pivot_data)
sns.heatmap(pivot_data, cmap='Blues', annot=True)
sns.heatmap(pivot_data, cmap='Blues', annot=True, fmt='d')
fig, (ax, cbar_ax) = plt.subplots(2)
print(fig)
print(ax)

grid_kws = {'height_ratios': (.85, .1), 'hspace': 0.4}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(pivot_data, ax=ax, cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'}, cmap='Blues', annot=True, fmt='d')
print(f)
