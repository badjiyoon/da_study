import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (10, 5)})


def function1(x):
    if x < 3:
        return 's'
    elif x < 5:
        return 'm'
    else:
        return 'l'


def function2(x):
    if x < 3:
        return 3
    elif x < 5:
        return 5
    else:
        return 7


df = sns.load_dataset('iris')
print(df.head())
print(df.isnull().sum())
# sns.scatterplot(data=df, x='sepal_width', y='sepal_length')
# hue를 이용한 분류
# sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='species')
print(df['petal_length'].unique())
df['petal_length2'] = df['petal_length'].apply(function1)
# sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='species', style='petal_length2')
df['petal_length3'] = df['petal_length'].apply(function2)
# sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='species', size='petal_length3')
# sns.relplot(data=df,  x='sepal_width', y='sepal_length', col='species')
sns.lmplot(data=df, x='sepal_width', y='sepal_length', hue='species')
plt.show()
