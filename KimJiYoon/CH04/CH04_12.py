import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize': (15, 5)})

df = sns.load_dataset('iris')
print(df.head())
# sns.distplot(df['sepal_length'])
# sns.distplot(df['sepal_length'], bins=15)
# hisplot
# sns.histplot(df['sepal_length'])
# 구간 개수 설정
# sns.histplot(df['sepal_length'], bins=20, kde=True)
# cout-> density
# sns.histplot(df['sepal_length'], bins=20, kde=True, stat='density')
# sns.histplot(df, bins=20, kde=True, stat='density', hue='species')
# sns.histplot(df, y='sepal_length', bins=20, kde=True, stat='density', hue='species')
# sns.histplot(df, y='sepal_length', x='sepal_width')
# 칼라바 추가
# sns.histplot(df, y='sepal_length', x='sepal_width', cbar=True)
# displot
# sns.displot(df['sepal_length'])
# sns.displot(data=df, x='sepal_length')
# sns.displot(data=df, x='sepal_length', height=5, aspect=3)
# sns.displot(data=df, y='sepal_length', bins=20, kde=True, stat='density', hue='species')
# sns.distplot(df, y='sepal_length', x='sepal_width', cbar=True)
# join plot
# sns.jointplot(data=df, x='sepal_width', y='sepal_length', kind='hist', cbar=True)
sns.jointplot(data=df, x='sepal_width', y='sepal_length', kind='hist', cbar=True, height=10)
plt.show()
