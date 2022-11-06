import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize': (10, 5)})

df = sns.load_dataset('titanic')
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.groupby(by='sex')['sex'].count())
print(df[['sex','class']])
print(df.groupby(by=['sex', 'class'])['sex'].count())
sns.countplot(data=df, x='sex', hue='class', palette='flare')
plt.show() 