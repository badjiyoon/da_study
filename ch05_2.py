
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df =sns.load_dataset('titanic')
df.head()

df.describe()

df.info()

train_df = df[:800]
test_df = df[800:]

train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df[['sex', 'survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)

train_df[['parch', 'survived']].groupby(['parch'], as_index=False).mean().sort_values(by='survived', ascending=False)

sns.histplot(data = train_df, x = 'age',bins = 20, hue = 'survived')

a = sns.FacetGrid(train_df, col='survived')
a.map(plt.hist, 'age', bins=20)

names = train_df.columns

train_df = train_df.drop(names[4:], axis=1)

train_df.head()

train_df.isnull().sum()
test_df.isnull().sum()

map_dict = {'femaie' : 0, 'maie' : 1}