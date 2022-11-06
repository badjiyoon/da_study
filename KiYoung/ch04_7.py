import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (10, 5)})


def functionA(x):
    if x < 2:
        return 's'
    elif x < 3:
        return 'm'
    else:
        return 'l'


def functionB(x):
    if x < 2:
        return 2
    elif x < 3:
        return 3
    else:
        return 4


df = sns.load_dataset('iris')
print(df.head())
print(df.isnull().sum())

sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='species')
print(df['petal_length'].unique())
df['petal_length2'] = df['petal_length'].apply(functionA)
df['petal_length3'] = df['petal_length'].apply(functionB)
sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='species', size='petal_length3')
sns.relplot(data=df,  x='sepal_width', y='sepal_length', col='species')
 