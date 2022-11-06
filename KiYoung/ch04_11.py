import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd
df = sns.load_dataset('iris')
sns.boxplot(data = df)
plt.subplots(figsize = (7, 8))
sns.boxplot(data = df)
sns.boxplot(data = df, orient = 'h')
plt.subplots(figsize = (10, 10))
sns.swarmplot(data = df)
plt.subplots(figsize = (10, 10))
sns.boxplot(data = df)
sns.swarmplot(data = df)
plt.subplots(figsize = (10, 10))
sns.boxplot(data = df)
sns.swarmplot(data = df, color = 'red')
sns.violinplot(data = df)
sns.stripplot(data = df)#산점도
plt.show()