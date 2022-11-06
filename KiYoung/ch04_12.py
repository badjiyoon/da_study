# 히스토그램

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15, 5)})
import pandas as pd
df = sns.load_dataset('iris')
sns.distplot(df['sepal_length'])
plt.show()
sns.distplot(df['sepal_length'], bins = 15)
plt.show()
sns.histplot(df['sepal_length'])
plt.show()
sns.histplot(df['sepal_length'], bins = 20)
plt.show()

sns.histplot(df['sepal_length'], bins = 20, kde = True)
plt.show()

sns.histplot(df['sepal_length'], bins = 20, kde = True, stat = 'density')
sns.histplot(df['sepal_length'], bins = 20, kde = True, stat = 'density', hue = 'species')
sns.histplot(data = df, x = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')

sns.histplot(data = df, y = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')
sns.histplot(data = df, y = 'sepal_length', x = 'sepal_width')
sns.histplot(data = df, y = 'sepal_length', x = 'sepal_width', cbar = True)
sns.displot(df['sepal_length'])
sns.displot(data = df, x = 'sepal_length')
sns.displot(data = df, x = 'sepal_length', height = 5, aspect = 3)
a = sns.displot(data = df, y = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')
sns.displot(data = df, y = 'sepal_length', x = 'sepal_width', cbar = True)
sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length')
sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length', kind = 'hist', cbar = True)

sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length', kind = 'hist', cbar = True, height = 10)
plt.show()